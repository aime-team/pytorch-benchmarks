#!/usr/bin/env python3

import torch
from random import Random
import torchvision
import numpy as np


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


def load_data(rank, args):
    """Loads the training data.
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                 ])
    # args.img_folder = '~/gpu_benchmark/fruits-360/Training'

    if args.img_folder:
        # Real data
        img_dataset = torchvision.datasets.ImageFolder(root=args.img_folder, transform=transforms)

    else:
        # Random images
        image_size = (3, 224, 224)
        num_images = 100000
        num_classes = 1000
        img_dataset = RandomDataset(image_size, num_images, num_classes, transform=transforms)
    if args.num_workers == -1:
        num_workers = args.num_gpus
    else:
        num_workers = args.num_workers
    val_data = ''
    if args.split_data != -1:
        torch.manual_seed(args.set_seed)
        ratio_map = [round(args.split_data * len(img_dataset)), round((1 - args.split_data) * len(img_dataset))]
        img_dataset, val_dataset = torch.utils.data.random_split(img_dataset, ratio_map)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.num_gpus, rank=rank)
        val_data = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=val_sampler
                                                )

    train_sampler = torch.utils.data.distributed.DistributedSampler(img_dataset, num_replicas=args.num_gpus, rank=rank)
    img_data = torch.utils.data.DataLoader(
                                        dataset=img_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        sampler=train_sampler
                                            )
    return img_data, val_data
