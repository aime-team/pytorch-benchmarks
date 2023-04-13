#!/usr/bin/env python3

import torch
import torchvision
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from PIL import Image
from data.bert_data_preprocessing import BertDataPreprocessing


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


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class MultiGpuData(object):
    def __init__(self, args):
        self.args = args
        self.total_steps_train = 0
        self.total_steps_eval = 0
        self.train_sampler = None
        self.set_seed()

    def set_seed(self):        
        if self.args.seed:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
        

    def init_distributed_dataloader(self, dataset, is_training):
        """Initializes the dataloader with distributed sampling.
        """
        if dataset:
            if self.args.distributed:
                sampler = DistributedSampler(dataset)
                if is_training:
                    batch_size = self.args.batch_size
                else:
                    batch_size = self.args.eval_batch_size
            else:
                sampler = RandomSampler(dataset)
                if is_training:
                    batch_size = self.args.global_batch_size
                else:
                    batch_size = self.args.global_eval_batch_size
            
            if not self.args.stress:
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=self.args.pin_memory,
                    sampler=sampler
                                        )
            else:
                dataloader = InfiniteDataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=self.args.pin_memory,
                    sampler=sampler
                                                )

        else:
            dataloader = None
            sampler = None

        return dataloader, sampler

    def get_train_dataloader(self):
        """Get the dataloader of the training data.
        """
        if not self.args.eval_only:
            train_data_loader, self.train_sampler = self.init_distributed_dataloader(self.train_dataset, True)
            if not self.args.stress:
                self.total_steps_train = len(train_data_loader)
            else:
                self.total_steps_train = np.inf

            return train_data_loader

    def get_eval_dataloader(self):
        """Get the dataloader of the validation data.
        """
        if self.args.eval:
            eval_data_loader, _ = self.init_distributed_dataloader(self.eval_dataset, False)
            self.total_steps_eval = len(eval_data_loader)
            return eval_data_loader


class MultiGpuImageData(MultiGpuData):
    """Data prepared for multi GPU training.
    """
    
    def __init__(self, args):
        super(MultiGpuImageData, self).__init__(args)

        self.train_dataset, self.eval_dataset = self.load_dataset()
        

    def load_imagenet_dataset(self):
        if not self.args.eval_only:
            train_dataset = torchvision.datasets.ImageNet(
                root=self.args.imagenet, transform=self.transforms['train'], split='train'
                                                        )
        else:
            train_dataset = None
        if self.args.eval:
            eval_dataset = torchvision.datasets.ImageNet(
                root=self.args.imagenet, transform=self.transforms['val'], split='val'
                                                        )
        else:
            eval_dataset = None
        if self.args.split_data != 1:
            ratio_map = [
                round(self.args.split_data * len(train_dataset)),
                round((1 - self.args.split_data) * len(train_dataset))
            ]
            train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, ratio_map)
        return train_dataset, eval_dataset

    def load_synthetic_dataset(self, is_training):
        mode_list = ['train', 'val']
        image_size = (3, 224, 224)
        num_classes = 1000
        dataset = RandomDataset(
            image_size, self.args.num_synth_data, num_classes, transform=self.transforms[mode_list[int(is_training)]]
                                        )
        return dataset


    def load_dataset(self):
        """Loads the dataset given in self.args. 
        Returns a tuple of the training dataset and the evaluation dataset
        """
        if self.args.imagenet:
            train_dataset, eval_dataset = self.load_imagenet_dataset()

        else:
            if self.args.train_image_folder:
                if not self.args.no_augmentation:
                    train_dataset = torchvision.datasets.ImageFolder(
                        root=self.args.train_image_folder, transform=self.transforms['train']
                                                                    )
                else:
                    train_dataset = torchvision.datasets.ImageFolder(
                        root=self.args.train_image_folder, transform=self.transforms['val']
                                                                    )
            elif not self.args.eval_only:
                train_dataset = self.load_synthetic_dataset(is_training=True)
            elif self.args.eval_only:
                train_dataset = None

            if self.args.eval_image_folder:
                eval_dataset = torchvision.datasets.ImageFolder(
                    root=self.args.eval_image_folder, transform=self.transforms['val']
                                                               )

            elif self.args.split_data != 1:
                ratio_map = [
                    round(self.args.split_data * len(train_dataset)), round((1 - self.args.split_data) * len(train_dataset))
                             ]
                train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, ratio_map)       
            elif self.args.eval:
                eval_dataset = self.load_synthetic_dataset(is_training=False)
            else:
                eval_dataset = None

        return train_dataset, eval_dataset

    def get_label_dict(self):
        """Get the lookup dictionary for the class labels.
        """
        if self.args.imagenet:
            lookup_file = Path.cwd() / 'data' / 'imagenet_label_lookup.npy'
            label_dict = np.load(lookup_file, allow_pickle='TRUE').item()
        else:
            if self.args.eval_image_folder:
                label_dict = self.eval_dataset.classes
            elif self.args.train_image_folder:
                label_dict = self.train_dataset.classes
            else:
                label_dict = None

        return label_dict

    def load_single_picture(self, picture_file):
        """Loads given single picture file and returns batch-normalized tensor.
        """
        with Image.open(picture_file) as img:
            image = self.transforms['val'](img).float()
        image = image.unsqueeze(0)
        return image

class MultiGpuResNetData(MultiGpuImageData):
        
    def __init__(self, args):
        self.transforms = self.get_transforms()
        super(MultiGpuResNetData, self).__init__(args)

    def get_transforms(self):

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
            ])
                            }

        return transforms
        

class MultiGpuBertData(MultiGpuData):
    """Data prepared for multi GPU training.
    """
    def __init__(self, args):
        super(MultiGpuBertData, self).__init__(args)
        self.args = args
        self.preprocessed_data = BertDataPreprocessing(self.args)
        self.train_dataset = self.load_train_dataset()
        self.eval_dataset = self.load_eval_dataset()

    def load_synthetic_dataset(self, is_training):
        
            all_input_ids_full = torch.randint(low=0, high=28988, size=[self.args.num_synth_data, self.args.max_seq_length], dtype=torch.long)
            mask_fill_with_zeros_from_random_idx = (torch.arange(self.args.max_seq_length) < torch.randint(low=int(self.args.max_seq_length/2), high=self.args.max_seq_length, size=[self.args.num_synth_data])[...,None])
            mask_fill_with_zeros_until_random_idx = (torch.randint(low=int(self.args.max_query_length/2), high=self.args.max_query_length, size=[self.args.num_synth_data])[...,None] < torch.arange(self.args.max_seq_length))
            all_input_ids = all_input_ids_full*mask_fill_with_zeros_from_random_idx
            all_segment_ids = torch.ones(size=[self.args.num_synth_data, self.args.max_seq_length], dtype=torch.long)*mask_fill_with_zeros_from_random_idx*mask_fill_with_zeros_until_random_idx
            all_input_mask = torch.ones(size=[self.args.num_synth_data, self.args.max_seq_length], dtype=torch.long)*mask_fill_with_zeros_from_random_idx
            if is_training:
                all_start_positions = torch.randint(low=0, high=self.args.max_seq_length, size=[self.args.num_synth_data], dtype=torch.long)
                all_end_positions = torch.randint(low=0, high=self.args.max_seq_length, size=[self.args.num_synth_data], dtype=torch.long)
                dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_start_positions, all_end_positions)
            else:
                all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
                dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_example_index)
            return dataset

    def load_train_dataset(self):
        if not self.args.eval_only:
            if self.args.synthetic_data:
                train_dataset = self.load_synthetic_dataset(True)
            else:
                all_input_ids = torch.tensor([f.input_ids for f in self.preprocessed_data.train_features], dtype=torch.long)           
                all_segment_ids = torch.tensor([f.segment_ids for f in self.preprocessed_data.train_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in self.preprocessed_data.train_features], dtype=torch.long)
                all_start_positions = torch.tensor([f.start_position for f in self.preprocessed_data.train_features], dtype=torch.long)
                all_end_positions = torch.tensor([f.end_position for f in self.preprocessed_data.train_features], dtype=torch.long)
                train_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_start_positions, all_end_positions)

            return train_dataset

    def load_eval_dataset(self):
        if self.args.eval:
            if self.args.synthetic_data:
                eval_dataset = self.load_synthetic_dataset(False)
            else:
                all_input_ids = torch.tensor([f.input_ids for f in self.preprocessed_data.eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in self.preprocessed_data.eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in self.preprocessed_data.eval_features], dtype=torch.long)
                all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
                eval_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_example_index)
            return eval_dataset

def load_data(args):
    if args.bert:
        return MultiGpuBertData(args)
    else:
        return MultiGpuResNetData(args)

