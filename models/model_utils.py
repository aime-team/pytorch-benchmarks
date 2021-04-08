#!/usr/bin/env python3

import torch
import torchvision
from pathlib import Path

CHECKPOINT_PATH = Path.cwd() / 'model_checkpoints'


class MultiGpuModel(torch.nn.Module):
    """Initialises model for multi gpu calculation.
    """
    def __init__(self, rank, model_name, num_gpus, precision, parallel, eval_mode):
        """Initialises the model and the training parameters.
        """
        super(MultiGpuModel, self).__init__()
        self.rank = rank
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.precision = precision
        self.parallel = parallel
        self.eval_mode = eval_mode

        self.model = self.get_model_from_torchvision()
        self.model = self.set_train_or_eval_mode(self.eval_mode)
        self.model = self.set_precision_mode(self.precision)
        self.model.cuda()#.to(memory_format=torch.contiguous_format)
        self.model = self.set_distribution_mode(self.parallel)

    def get_model_from_torchvision(self):
        """Loads model from torchvision to self.model and sets attributes.
        """
        try:
            model = getattr(torchvision.models, self.model_name)(pretrained=False)
        except AttributeError:
            sys.exit(f'There is no model with the name {self.model_name} in torchvision.\n'
                     f'The following models are available for benchmark:\n'
                     f'{MultiGpuModel.get_available_models()}')
        return model

    def set_train_or_eval_mode(self, eval_mode):
        """Set the training / evaluation mode for the model.
        If the argument eval_mode is True, the model will get optimized for evaluation. If False for training.
        """
        if eval_mode:
            model = self.model.eval()
        else:
            model = self.model.train()
        return model

    def set_precision_mode(self, precision):
        """Set the precision of the model to half or float precision depending on given string.
        """
        #self.model = getattr(self.model, self.precision)()
        if precision == 'half':
            model = network_to_half(self.model)
        else:
            model = self.model
        return model

    def set_distribution_mode(self, parallel):
        """Set the distribution mode for multi gpu training / evaluation.
        If argument parallel is True, DataParallel is used. If False DistributedDataParallel.
        """
        if parallel:
            model = torch.nn.parallel.DataParallel(self.model, device_ids=list(range(num_gpus)), dim=0)
        else:
            model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        return model

    def forward(self, data):
        """Connects self.model with the module.
        """
        return self.model(data)

    def save_model(self, optimizer, epoch, data_name):
        """Saves model checkpoint to file.
        """
        if not CHECKPOINT_PATH.is_dir():
            CHECKPOINT_PATH.mkdir()
        file = CHECKPOINT_PATH / f'{self.model_name}_epoch_{epoch}_{data_name}.pt'
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
    def check_saved_checkpoint_epoch(model_name, data_name):
        """Returns maximum available epoch to load pretrained model state.
        """
        epoch_list = []
        for filename in CHECKPOINT_PATH.rglob(f'{model_name}_epoch_*_{data_name}.pt'):
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

