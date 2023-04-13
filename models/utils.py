#!/usr/bin/env python3

import torch
import torchvision
import torch.distributed as dist
import sys
from utils.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from models.optimizer import BertAdam, Lamb
import datetime

class MultiGpuModel(torch.nn.Module):
    """Initialises model for multi gpu calculation.
    """
    def __init__(self, rank, args):
        """Initialises the model and the training parameters.
        """
        super(MultiGpuModel, self).__init__()
        torch.cuda.set_device(rank)
        self.device = torch.device(args.device)
        self.rank = rank
        self.args = args
        self.model = self.init_model()
        self.set_precision_mode(self.args.precision)        
        self.set_distribution_mode(self.args.distribution_mode)
        self.optimizer = self.init_optimizer()
        self.criterion = self.init_loss_function()
        self.scheduler = self.init_scheduler()
        self.scaler = self.init_scaler()
        self.set_seed()
        self.load_checkpoint()
        self.do_pytorch2_optimizations()


    def init_model(self):
        return self.get_model_from_torchvision().to(self.device)

    def get_weights(self):
        if self.args.pretrained:
            if self.args.model == 'resnet50':
                weights = torchvision.models.ResNet50_Weights.DEFAULT
            else:
                sys.exit('Weights for this model not implemented yet')
        else:
            weights = None
      

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

    def init_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.auto_mixed_precision)
        return scaler

    def init_loss_function(self, ignore_index=-100):
        """Initializes the loss function CrossEntropyLoss.
        """
        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        criterion.to(self.args.device)
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
                    world_size=self.args.num_gpus, init_method='env://'
                                        )
            except ValueError:
                if self.rank == 0:
                    print(
                        f'There is no backend called {self.args.process_group_backend}. '
                        f'Use nccl (default) or gloo.\n'
                        )
                sys.exit(0)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], find_unused_parameters=self.args.find_unused_parameters)
        elif distribution_mode == 2:
            self.model = torch.nn.parallel.DataParallel(self.model, device_ids=list(range(self.args.num_gpus)), dim=0)
        return True

    def do_pytorch2_optimizations(self):
        if self.args.compile:                    
            torch.set_float32_matmul_precision('high')
            self.model = torch.compile(self.model)

    def forward(self, *input):
        """Connects self.model(input) to the class.
        """
        return self.model(*input)

    def save_checkpoint(self, epoch, data_name):
        """Saves model checkpoint on given epoch with given data name.
        """
        if self.args.dist_optim or self.args.dist_optim_190:
            dist.barrier()
            for gpu_id in range(self.args.num_gpus):
                self.optimizer.consolidate_state_dict(to=gpu_id)
        if not self.args.checkpoint_folder.parent.is_dir():
            self.args.checkpoint_folder.parent.mkdir()        
        if not self.args.checkpoint_folder.is_dir():
            self.args.checkpoint_folder.mkdir()
        file = self.args.checkpoint_folder / f'{self.args.model}_{data_name}_epoch_{epoch}.pt'
        if not file.is_file():
            file.touch()
        if self.args.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            file
                   )
        return True


    def load_checkpoint(self):
        """Loads model checkpoint from file to the GPU with given rank.
        """
        if self.args.load_from_epoch != 0:
            if self.rank == 0:
                print(f'Load checkpoint from {self.args.checkpoint_file}...', end='', flush=True)
            map_location = {'cuda:0': f'cuda:{self.rank}'}
            checkpoint = torch.load(self.args.checkpoint_file, map_location=map_location)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.rank == 0:
                print('Done')
        elif self.args.pretrained and not self.args.model == 'resnet50':
            if self.rank == 0:
                print(f'Load checkpoint from {self.args.checkpoint_file}...', end='', flush=True)
            self.load_downloaded_checkpoint()
            if self.rank == 0:
                print('Done')


    def load_downloaded_checkpoint(self):
        if self.args.pretrained:
            map_location = {'cuda:0': f'cuda:{self.rank}'}
            state_dict = torch.load(self.args.checkpoint_file, map_location=map_location)
            try:
                self.model.load_state_dict(state_dict, strict=False)
            except RuntimeError:
                dist.init_process_group(backend=self.args.process_group_backend, rank=self.rank, 
                    world_size=self.args.num_gpus, init_method='env://')
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], find_unused_parameters=True)
                self.model.load_state_dict(state_dict, strict=False)
                self.model = self.model.module


    def set_seed(self):
            
        if self.args.seed:
            #np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

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


    def average_gradients(self):
        """Calculates the average of the gradients over all gpus.
        """
        for param in self.model.parameters():
            dist.barrier()
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.args.num_gpus
        return True

class MultiGpuBertModel(MultiGpuModel):
    def __init__(self, rank, args):
        """Initialises the model and the training parameters.
        """
        super(MultiGpuBertModel, self).__init__(rank, args)

    def init_model(self):
        import models.bert

        config = models.bert.BertConfig.from_dict(self.args.bert_config_dict)
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)

        model = models.bert.BertForQuestionAnswering(config)
        return model.to(self.device)

    def init_optimizer(self):
        """Initializes the optimizer and the scheduler.
        """
        
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        """if self.args.auto_mixed_precision:
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                bias_correction=False)
            scheduler = LinearWarmUpScheduler(optimizer, warmup=self.args.warmup_proportion, total_steps=num_train_optimization_steps)
        """
        optimizer = BertAdam(optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    warmup=self.args.warmup_proportion,
                    t_total=self.args.num_train_optimization_steps)
        
        return optimizer

    def do_backpropagation(self, model_output, model_target_output):

        start_logits, end_logits = model_output
        start_positions, end_positions = model_target_output
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = self.init_loss_function(ignore_index=ignored_index)
        
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        if self.args.num_gpus > 1:
            loss = loss.mean() 
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def do_batch_processing(self, batch):
        batch = tuple(element.to(self.device, non_blocking=self.args.pin_memory) for element in batch) 
        model_input, model_target_output = batch[0:3], batch[3:5] 
        return model_input, model_target_output


class MultiGpuImageModel(MultiGpuModel):
    def __init__(self, rank, args):
        """Initialises the model and the training parameters.
        """
        super(MultiGpuImageModel, self).__init__(rank, args)

    def get_model_from_torchvision(self):
        """Loads model from torchvision to self.model and sets attributes.
        """
        try:
            weights = self.get_weights()
            
            model = getattr(torchvision.models, self.args.model)(weights=weights)
        except AttributeError:
            if self.rank == 0:
                print(
                    f'There is no model with the name {self.args.model} in torchvision.\n'
                    f'The following models are available for benchmark:\n'
                    f'{torchvision.models.list_models()}'
                    )
            sys.exit(0)
        return model


    def do_backpropagation(self, model_output, model_target_output):
        loss = self.criterion(model_output, model_target_output)
        self.scaler.scale(loss).backward()
        if self.args.average_gradients and self.args.distributed:
            self.average_gradients()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss

    def do_batch_processing(self, batch):
        batch = tuple(element.to(self.device, non_blocking=self.args.pin_memory) for element in batch) 
        model_input, label = batch
        return  (model_input,),  label
        

    def predict_label_for_single_picture(self):
        """Evaluates a single picture given in args.pred_pic_label, prints and returns the predicted label.
        """
        if self.args.distributed:
            dist.init_process_group(backend=self.args.process_group_backend, rank=self.rank, 
                    world_size=self.args.num_gpus, init_method='env://')
        from data.data import MultiGpuData
        img_data = MultiGpuData(self.args)
        if self.args.load_from_epoch == -1:
            self.args.load_from_epoch = MultiGpuModel.check_saved_checkpoint_epoch(self.args.model, img_data.data_name, self.args.checkpoint_folder)
        image = img_data.load_single_picture(self.args.pred_pic_label)
        #self.model.model = self.model.load_checkpoint(
        #    self.args.load_from_epoch, img_data.data_name, 0
        #                               )
        output = self.model(image)
        label_dict = img_data.get_label_dict()
        prediction = label_dict[int(torch.argmax(output, 1))]
        dist.destroy_process_group()
        sys.exit(
            f'\nThe predicted class calculated by {self.args.model} trained by {img_data.data_name} '
            f'until epoch {self.args.load_from_epoch} is: {prediction}\n'
                 )  



class MultiGpuPerceiverModel(MultiGpuModel):
    def __init__(self, rank, args):
        """Initialises the model and the training parameters.
        """
        super(MultiGpuPerceiverModel, self).__init__(rank, args)

    def init_model(self):
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
        return model.to(self.device)

    def do_backpropagation(self, model_output, model_target_output):
        loss = self.criterion(model_output, model_target_output)
        self.scaler.scale(loss).backward()
        if self.args.average_gradients and self.args.distributed:
            self.average_gradients()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss

    def do_batch_processing(self, batch):
        batch = tuple(element.to(self.device, non_blocking=self.args.pin_memory) for element in batch) 
        model_input, model_target_output = batch
        model_target_output = model_target_output.transpose(1,3).transpose(1,2)
        return model_input, model_target_output

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

def init_multi_gpu_model(rank, args):
    if args.bert:
        model = MultiGpuBertModel(rank, args)
    elif args.model == 'perceiver':
        model = MultiGpuPerceiverModel(rank, args)
    else:
        model = MultiGpuImageModel(rank,args)
    return model
