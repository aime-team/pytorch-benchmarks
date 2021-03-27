#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import platform
import datetime
import time
import sys
import argparse
import numpy as np
import subprocess

torch.backends.cudnn.benchmark = True

class RandomDataset(torch.utils.data.Dataset):
    """Creates Random images"""

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


def load_flags():
    """Parses arguments."""

    parser = argparse.ArgumentParser(description='PyTorch Benchmarking')

    parser.add_argument(
        '--warm_up_steps', '-w', type=int, default=10, required=False,
        help="Number of warm up steps in every epoch. Warm up steps will not taken into account"
                        )
    parser.add_argument('--num_epochs', '-n', type=int, default=10, required=False, help="Number of epochs")
    parser.add_argument('--batch_size', '-b', type=int, default=64, required=False, help='Batch size')
    parser.add_argument('--num_gpus', '-g', type=int, default=1, required=False, help='Number of gpus used for training')
    parser.add_argument(
        '--gpu_ids', '-i', type=int, nargs='+', required=False,
        help='IDs of used GPUs for training. If not given, range(num_gpus - 1) is used.'
                        )
    parser.add_argument('--model', '-m', type=str, default='resnet50', required=False, help='Model used for training')
    parser.add_argument('--use_fp16', '-f', action='store_true', required=False, help='Use half precision')
    parser.add_argument(
        '--img_folder', '-imf', type=str, required=False,
        help='Destination of training images. If not given, random data will be used.'
                        )
    parser.add_argument(
        '--num_workers', '-nw', type=int, default=-1, required=False,
        help='Number of workers for the dataloader. If not given num_gpus is used.'
                        )

    _args = parser.parse_args()
    _args.batch_size *= _args.num_gpus
    return _args


def make_info_text(_precision):
    """Makes info text about the device and the OS shown at the beginning of the benchmark and in the protocol."""

    gpu_name_dict = {}
    _info_text = f'OS: {platform.uname().system}, {platform.uname().release}\n'\
                 f'Device-name: {platform.uname().node}\n'\
                 f'{args.num_gpus} GPU(s) used for benchmark:\n'
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = range(args.num_gpus)

    for gpu_id in gpu_ids:
        gpu_name = str(torch.cuda.get_device_name(gpu_id))
        gpu_name_dict[gpu_id] = gpu_name
        _info_text += f'{gpu_id}: {gpu_name}\n'

    _info_text += f'Available GPUs on device: {torch.cuda.device_count()}\n'\
                  f'Cuda-version: {torch.version.cuda}\n'\
                  f'Cudnn-version: {torch.backends.cudnn.version()}\n'

    cpu_name = 'unknown'
    for line in subprocess.check_output("lscpu", shell=True).strip().decode().split('\n'):
        if 'Modellname' in line or 'Model name' in line:
            cpu_name = line.split(':')[1].strip()

    _info_text += f'CPU: {cpu_name}\n'\
                  f'Used model: {args.model}\n'\
                  f'Batch size: {args.batch_size}\n'\
                  f'Precision: {_precision}\n'

    if args.img_folder:
        _info_text += f'Training data: {args.img_folder}\n'
    else:
        _info_text += f'Training data: Random images\n'

    _info_text += f'Warm up steps: {args.warm_up_steps}\n'
    _now = datetime.datetime.now()
    start_time = _now.strftime('%Y/%m/%d %H:%M:%S')

    _info_text += f'Benchmark start : {start_time}\n\n'
    return _info_text


def make_protocol(_img_per_sec_dict, _info_text):
    """Writes benchmark results in a textfile and calculates the mean.
    Takes the images_per_sec_dict and the infotext as arguments. Returns the mean of images per sec.
    """
    with open(f'{args.num_gpus}_{str(torch.cuda.get_device_name(0))}_{args.batch_size}.txt', 'w') as protocol:
        protocol.write(_info_text)
        for key in _img_per_sec_dict:
            protocol.write(f'Epoch: {key[0]}, Step: {key [1]}, Images per second: {_img_per_sec_dict[key]}\n')

        _mean_img_per_sec = np.array(list(_img_per_sec_dict.values())).mean()
        protocol.write(f'\nMean images per second: {_mean_img_per_sec}\n')

        _now = datetime.datetime.now()
        _end_time = _now.strftime('%Y/%m/%d %H:%M:%S')
        protocol.write(f'Benchmark end : {_end_time}\n')
        return _mean_img_per_sec


def load_data():
    """Loads the training data."""
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
    _img_data = torch.utils.data.DataLoader(
        img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
                                            )
    _images, _labels = next(iter(_img_data))
    return _img_data


def load_model(_precision):
    """Loads model from torchvision."""
    try:
        _model = getattr(torchvision.models, args.model)(pretrained=False)
    except AttributeError:
        sys.exit(f'There is no model with the name {args.model} in torchvision.\n'
                 f'The following models are available for benchmark:\n'
                 f'{get_available_models()}')
    _model = getattr(_model, _precision)()
    if args.num_gpus > 1:
        if args.gpu_ids:
            _model = torch.nn.parallel.DataParallel(_model, device_ids=args.gpu_ids, dim=0)
        else:
            _model = torch.nn.parallel.DataParallel(_model, device_ids=list(range(args.num_gpus)), dim=0)
    _model = _model.train()
    _model = _model.cuda()
    return _model


def get_available_models():
    model_class_list = [
        torchvision.models.densenet,
        torchvision.models.mnasnet,
        torchvision.models.mobilenet,
        torchvision.models.shufflenetv2,
        torchvision.models.resnet,
        torchvision.models.squeezenet,
        torchvision.models.vgg
                        ]
    model_dict = {'AlexNet': 'alexnet', 'GoogleNet': 'googlenet'}
    all_models_str = ''
    for model_class in model_class_list:
        model_dict[f'{model_class.__all__[0]}'] = ', '.join(model_class.__all__[1:])
    for model_class in model_dict:
        all_models_str += f'{model_class}: {model_dict[model_class]}\n'
    return all_models_str


def train(_model, _img_data, _precision='float'):
    calc_every = 10
    total_step = len(_img_data)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = torch.optim.SGD(_model.parameters(), lr=learning_rate, momentum=0.9)
    durations = []
    start = time.time()
    for epoch in range(1, args.num_epochs + 1):
        print(f'Epoch {epoch}')
        for step, (_data, _label) in enumerate(_img_data):
            _data, _label = _data.cuda(), _label.cuda()
            optimizer.zero_grad()
            _data = getattr(_data, _precision)()
            outputs = _model(_data)
            loss = criterion(outputs, _label)
            loss.backward()
            optimizer.step()
            end = time.time()
            durations.append((end - start))
            start = end
            if step > 0 and (step + 1) % calc_every == 0 or (step + 1) == total_step:

                duration = sum(durations) / len(durations)
                durations = []
                img_per_sec = args.batch_size / duration
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{(step +1)}/{total_step}], Loss: {loss.item():.4f}, '
                      f'Images per second: {img_per_sec}')
                if step >= args.warm_up_steps + 1:
                    img_per_sec_dict[(epoch, step)] = img_per_sec
    return img_per_sec_dict


if __name__ == '__main__':
    learning_rate = 1e-3
    img_per_sec_dict = {}
    args = load_flags()

    if args.use_fp16:
        precision = 'half'
    else:
        precision = 'float'

    info_text = make_info_text(precision)
    print(info_text)

    img_data = load_data()
    model = load_model(precision)
    try:
        train(model, img_data, precision)
    except KeyboardInterrupt:
        if len(list(img_per_sec_dict.values())) == 0:
            sys.exit('Cancelled in warm up stage')
        mean_img_per_sec = make_protocol(img_per_sec_dict, info_text)
        sys.exit(f'\nMean images per second: {mean_img_per_sec}\n'
                 f'KeyboardInterrupt')

    mean_img_per_sec = make_protocol(img_per_sec_dict, info_text)
    print(f'\nMean images per second: {mean_img_per_sec}\n')
    now = datetime.datetime.now()
    end_time = now.strftime('%Y/%m/%d %H:%M:%S')
    print(f'Benchmark end : {end_time}')
