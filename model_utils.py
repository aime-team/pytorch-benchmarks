#!/usr/bin/env python3

import torch
import torchvision
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP

CHECKPOINT_PATH = Path.cwd() / 'model_checkpoints'

class MultiGpuModel(object):
    """Initialises model for multi gpu calculation.
    """
    def __init__(self, rank, model_name, num_gpus, precision, parallel_mode):
        """Loads model from torchvision to self.model and sets attributes.
        """
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.precision = precision
        try:
            self.model = getattr(torchvision.models, self.model_name)(pretrained=False)
        except AttributeError:
            sys.exit(f'There is no model with the name {self.model_name} in torchvision.\n'
                     f'The following models are available for benchmark:\n'
                     f'{MultiGpuModel.get_available_models()}')

        #self.model = getattr(self.model, self.precision)()
        if self.precision == 'half':
            self.model = network_to_half(self.model)
        self.model = self.model.train()
        self.model.cuda().to(memory_format=torch.contiguous_format)

        if parallel_mode:
            self.model = torch.nn.parallel.DataParallel(self.model, device_ids=list(range(num_gpus)), dim=0)
        else:
            self.model = DDP(self.model, device_ids=[rank])

    def save_model(self, optimizer, epoch):
        """Saves model state to file.
        """
        if not CHECKPOINT_PATH.is_dir():
            CHECKPOINT_PATH.mkdir()
        file = CHECKPOINT_PATH / f'{self.model_name}_epoch_{epoch}.pt'
        if not file.is_file():
            file.touch()
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            file
        )

    def load_model(self, optimizer, epoch, map_location):
        """Loads model state from file.
        """
        file = CHECKPOINT_PATH / f'{self.model_name}_epoch_{epoch}.pt'
        checkpoint = torch.load(file, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self.model, optimizer

    @staticmethod
    def check_pretrained_model_epoch(model_name):
        """Returns maximum available epoch to load pretrained model state.
        """
        epoch_list = []
        for filename in CHECKPOINT_PATH.rglob(f'{model_name}_epoch_*.pt'):
            epoch = filename.name.split('_')[2].strip('.pt')
            epoch_list.append(int(epoch))
        return max(epoch_list)

    @staticmethod
    def get_available_models():
        """Returns string with all available models in torchvision.
        """
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


class ToFp16(torch.nn.Module):
    """Utility module that implements::
        def forward(self, input):
            return input.half()
    """
    def __init__(self):
        super(ToFp16, self).__init__()

    def forward(self, input):
        return input.half()


def bn_convert_float(module):
    """
    Utility function for network_to_half().
    Retained for legacy purposes.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
        module.float()
    for child in module.children():
        bn_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    Retained for legacy purposes. It is recommended to use FP16Model.
    """
    return torch.nn.Sequential(ToFp16(), bn_convert_float(network.half()))

