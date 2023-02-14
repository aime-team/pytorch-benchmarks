#!/usr/bin/env python3

import torch
import torchvision
import torch.distributed as dist
import sys
from utils.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from utils.optimizer import Lamb


class MultiGpuModel(torch.nn.Module):
    """Initialises model for multi gpu calculation.
    """
    def __init__(self, rank, args):
        """Initialises the model and the training parameters.
        """
        super(MultiGpuModel, self).__init__()
        torch.cuda.set_device(rank)
        self.rank = rank
        self.args = args
        self.num_gpus = args.num_gpus
        self.model_name = args.model
        self.checkpoint_folder = args.checkpoint_folder
        self.model = self.load_model()
        self.set_precision_mode(args.precision)        
        self.model.cuda(rank)
        self.set_distribution_mode(args.distribution_mode)
        self.optimizer = self.init_optimizer()
        self.criterion = self.init_loss_function()
        self.scheduler = self.init_scheduler()
        self.do_pytorch2_optimizations()

    def get_model_from_torchvision(self):
        """Loads model from torchvision to self.model and sets attributes.
        """
        try:
            model = getattr(torchvision.models, self.model_name)(weights=self.args.pretrained)
        except AttributeError:
            if self.rank == 0:
                print(
                    f'There is no model with the name {self.model_name} in torchvision.\n'
                    f'The following models are available for benchmark:\n'
                    f'{self.get_available_models()}'
                    )
            sys.exit(0)
        return model

    def load_model(self):
        if self.model_name == 'perceiver':
            from models.perceiver import Perceiver

            model = Perceiver(
                input_channels = 3,          # number of channels for each token of the input
                input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
                num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
                depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                            #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = 512,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                num_classes = 1000,          # output number of classes
                attn_dropout = 0.,
                ff_dropout = 0.,
                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2      # number of self attention blocks per cross attention
            )
        else:
            model = self.get_model_from_torchvision()
        #model = torch.compile(model)

        return model
        

    def init_optimizer(self):
        """Initializes the optimizer.
        """
        if self.args.optimizer == 'SGD':
            if self.args.dist_optim_190:
                # Only for Pytorch > 1.9.0
                optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
                    self.model.parameters(), optimizer_class=torch.optim.SGD, lr=self.args.learning_rate, momentum=0.9
                                                                            )
            elif self.args.dist_optim:
                optimizer = ZeroRedundancyOptimizer(
                    self.model.parameters(), optimizer_class=torch.optim.SGD, lr=self.args.learning_rate, momentum=0.9
                                                    )
            else:
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9,
                                    weight_decay=1e-4)
                
        elif self.args.optimizer == 'Lamb':
            optimizer = Lamb(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        else:
            if self.rank == 0:
                print(
                    f'There is no optimizer called {self.args.optimizer}. '
                    f'Use "SGD" or "Lamb".\n'
                      )
            sys.exit(0)
        
        return optimizer

    def init_scheduler(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_lr, gamma=1/self.args.lr_decay_factor)
        return scheduler

    def init_loss_function(self):
        """Initializes the loss function CrossEntropyLoss.
        """
        criterion = torch.nn.CrossEntropyLoss()
        criterion.cuda(self.rank)
        return criterion

    def set_precision_mode(self, precision):
        """Set the precision of the model to half or float precision depending on given string.
        """
        self.model = getattr(self.model, precision)()
        if precision == 'half':
            self.model = network_to_half(self.model)
        else:
            self.model = self.model
        return True

    def set_distribution_mode(self, distribution_mode):
        """Set the distribution mode for multi gpu training / evaluation model by given argument.
        1 stands for DistributedDataParallel, 2 for DataParallel and 0 for Single GPU model.
        """
        if distribution_mode == 0:
            self.model = self.model
        elif distribution_mode == 1:
            try:
                dist.init_process_group(
                    backend=self.args.process_group_backend, rank=self.rank, 
                    world_size=self.num_gpus, init_method='env://'
                                        )
            except ValueError:
                if self.rank == 0:
                    print(
                        f'There is no backend called {self.args.process_group_backend}. '
                        f'Use gloo (default) or nccl.\n'
                        )
                sys.exit(0)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        elif distribution_mode == 2:
            self.model = torch.nn.parallel.DataParallel(self.model, device_ids=list(range(self.num_gpus)), dim=0)
        return True

    def do_pytorch2_optimizations(self):
        if self.args.pytorch2:                    
            torch.set_float32_matmul_precision('high')
            self.model = torch.compile(self.model)

    def forward(self, data):
        """Connects self.model(input) to the class.
        """
        return self.model(data)

    def save_model(self, epoch, data_name):
        """Saves model checkpoint on given epoch with given data name.
        """
        if self.args.dist_optim or self.args.dist_optim_190:
            dist.barrier()
            for gpu_id in range(self.args.num_gpus):
                self.optimizer.consolidate_state_dict(to=gpu_id)
        if not self.checkpoint_folder.parent.is_dir():
            self.checkpoint_folder.parent.mkdir()        
        if not self.checkpoint_folder.is_dir():
            self.checkpoint_folder.mkdir()
        file = self.checkpoint_folder / f'{self.model_name}_{data_name}_epoch_{epoch}.pt'
        if not file.is_file():
            file.touch()
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            file
                   )
        return True

    def load_pretrained_model(self, epoch, data_name, rank):
        """Loads model state from file to the GPU with given rank.
        """
        if self.args.pretrained:
            pass
        else:
            if self.args.distributed:
                dist.barrier()
            map_location = {'cuda:0': f'cuda:{rank}'}
            file = self.checkpoint_folder / f'{self.model_name}_{data_name}_epoch_{epoch}.pt'
            checkpoint = torch.load(file, map_location=map_location)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return True

    @staticmethod
    def check_saved_checkpoint_epoch(model_name, data_name, checkpoint_folder):
        """Returns maximum available epoch to load pretrained model state.
        """
        epoch_list = []
        for filename in checkpoint_folder.rglob(f'{model_name}_{data_name}_epoch_*.pt'):
            epoch = filename.name.strip('.pt').split('_')[-1]
            epoch_list.append(int(epoch))
        if epoch_list:
            return max(epoch_list)
        else:
            print(f'No checkpoint found in {checkpoint_folder}. Checkpoint filenames have the pattern {model_name}_{data_name}_epoch_*.pt')
            sys.exit()


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

    def predict_label_for_single_picture(self):
        """Evaluates a single picture given in args.pred_pic_label, prints and returns the predicted label.
        """
        if self.args.distributed:
            dist.init_process_group(backend='gloo', rank=0, world_size=self.num_gpus, init_method="env://")
        from data.data import MultiGpuData
        img_data = MultiGpuData(self.args)
        if self.args.load_from_epoch == -1:
            self.args.load_from_epoch = MultiGpuModel.check_saved_checkpoint_epoch(self.args.model, img_data.data_name, self.args.checkpoint_folder)
        image = MultiGpuData.load_single_picture(self.args.pred_pic_label)
        #self.model.model = self.model.load_pretrained_model(
        #    self.args.load_from_epoch, img_data.data_name, 0
        #                               )
        output = self.model(image)
        prediction = img_data.label_dict[int(torch.argmax(output, 1))]
        dist.destroy_process_group()
        sys.exit(
            f'\nThe predicted class calculated by {self.args.model} trained by {img_data.data_name} '
            f'until epoch {self.args.load_from_epoch} is: {prediction}\n'
                 )  

    def average_gradients(self):
        """Calculates the average of the gradients over all gpus.
        """
        for param in self.model.parameters():
            dist.barrier()
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.num_gpus
        return True


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
