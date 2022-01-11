#!/usr/bin/env python3

import torch
from random import Random
import torchvision
import numpy as np
from pathlib import Path
from PIL import Image


class RandomDataset(torch.utils.data.Dataset):
    """Creates Random images.
    """
    def __init__(self, img_size, length, _num_classes, transform=None):
        super().__init__()
        self.len = length
        self.data = torch.randn(img_size)
        self.num_classes = _num_classes
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        label = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        self.data = torchvision.transforms.ToPILImage()(self.data)
        if self.transform is not None:
            self.data = self.transform(self.data)
        return self.data, label


class Partition(object):
    """ Dataset-like object, but only access a subset of it.
    """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chunks.
    """
    def __init__(self, data, ratio=0.9, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [idx for idx in range(data_len)]
        rng.shuffle(indexes)

        for frac in [ratio, 1 - ratio]:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class MultiGpuData(object):
    """Data prepared for multi GPU training.
    """
    transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}
    def __init__(self, args):
        self.args = args
        self.data_name = MultiGpuData.make_data_name(args)
        self.train_dataset, self.val_dataset = self.load_dataset()
        self.label_dict = self.get_label_dict()

    def load_dataset(self):
        """Loads the dataset given in self.args. 
        Returns a tuple of the training dataset and the evaluation dataset
        """
        if self.args.imagenet:
            train_dataset = torchvision.datasets.ImageNet(
                root=self.args.imagenet, transform=MultiGpuData.transforms['train'], split='train'
                                                          )
            val_dataset = torchvision.datasets.ImageNet(
                root=self.args.imagenet, transform=MultiGpuData.transforms['val'], split='val'
                                                        )
            if self.args.split_data != 1:
                torch.manual_seed(self.args.set_seed)
                ratio_map = [
                    round(self.args.split_data * len(train_dataset)),
                    round((1 - self.args.split_data) * len(train_dataset))
                ]
                train_dataset, _ = torch.utils.data.random_split(train_dataset, ratio_map)

        else:
            if self.args.train_folder:
                train_dataset = torchvision.datasets.ImageFolder(
                    root=self.args.train_folder, transform=MultiGpuData.transforms['train']
                                                                 )
            else:
                image_size = (3, 224, 224)
                num_images = self.args.num_images
                num_classes = 1000
                train_dataset = RandomDataset(
                    image_size, num_images, num_classes, transform=MultiGpuData.transforms['train']
                                              )
                label_dict = [i for i in range(num_classes)]
            if self.args.val_folder:
                val_dataset = torchvision.datasets.ImageFolder(
                    root=self.args.val_folder, transform=MultiGpuData.transforms['val']
                                                               )            
                label_dict = val_dataset.classes
                if self.args.split_data != 1:
                    torch.manual_seed(self.args.set_seed)
                    ratio_map = [
                        round(self.args.split_data * len(train_dataset)), round((1 - self.args.split_data) * len(train_dataset))
                                 ]
                    train_dataset, _ = torch.utils.data.random_split(train_dataset, ratio_map)
            elif self.args.split_data != 1:
                torch.manual_seed(self.args.set_seed)
                ratio_map = [
                    round(self.args.split_data * len(train_dataset)), round((1 - self.args.split_data) * len(train_dataset))
                             ]
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, ratio_map)       
            else:
                val_dataset = ''

        return train_dataset, val_dataset

    def init_distributed_dataloader(self, dataset, rank):
        """Initialises the dataloader with distributed sampling for the gpu with given rank.
        """
        if dataset:
            if self.args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                shuffle=(sampler is None),
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                sampler=sampler
                                                    )
        else:
            dataloader = ''
            sampler = ''

        return dataloader, sampler

    def get_train_dataloader(self, rank):
        """Get the dataloader of the training data for the GPU with given rank.
        """
        return self.init_distributed_dataloader(self.train_dataset, rank)

    def get_val_dataloader(self, rank):
        """Get the dataloader of the validation data for the GPU with given rank.
        """
        return self.init_distributed_dataloader(self.val_dataset, rank)

    def get_label_dict(self):
        """Get the lookup dictionary for the class labels.
        """
        if self.args.imagenet:
            lookup_file = Path.cwd() / 'data' / 'imagenet_label_lookup.npy'
            label_dict = np.load(lookup_file, allow_pickle='TRUE').item()
        else:
            if self.args.val_folder:
                label_dict = self.val_dataset.classes
            elif self.args.train_folder:
                label_dict = self.train_dataset.classes
            else:
                label_dict = [i for i in range(len(self.train_dataset))]

        return label_dict

    @staticmethod
    def make_data_name(args):
        """Returns a string with the name of the training/evaluation datasets.
        """
        if args.imagenet:
            data_name  = 'ImageNet'
        elif args.train_folder:
            data_name_list = args.train_folder.strip('/').split('/')
            data_name  = f'{data_name_list[-2]}_{data_name_list[-1]}'
        else:
            data_name = 'Random data'
        return data_name

    @staticmethod
    def load_single_picture(picture_file):
        """Loads given single picture file and returns batch-normalized tensor.
        """
        with Image.open(picture_file) as img:
            image = MultiGpuData.transforms['val'](img).float()
        image = image.unsqueeze(0)
        return image
