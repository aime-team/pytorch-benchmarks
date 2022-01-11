import argparse
import sys
import os
import torch
from pathlib import Path
from data.data import MultiGpuData
from models.utils import MultiGpuModel


def load_flags():
    """Parses arguments and sets global parameters.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    parser = argparse.ArgumentParser(
        description='PyTorch Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument(
        '-w',  '--warm_up_steps', type=int, default=10, required=False,
        help="Number of warm up steps in every epoch. Warm up steps will not taken into account"
                        )
    parser.add_argument(
        '-ne', '--num_epochs', type=int, default=10, required=False, help='Number of total epochs. '
                        )
    parser.add_argument(
        '-b', '--batch_size', type=int, default=64, required=False, help='Batch size. Default: 64'
                        )
    parser.add_argument(
        '-ng', '--num_gpus', type=int, default=1, required=False, help='Number of gpus used for training'
                        )
    parser.add_argument(
        '-m', '--model', type=str, default='resnet50', required=False,
        help='Use different model from torchvision. Default: resnet50'
                        )
    parser.add_argument(
        '-f', '--use_fp16', action='store_true', required=False, help='Use half precision'
                        )
    parser.add_argument(
        '-dp', '--parallel', action='store_true', required=False, help='Use DataParallel for Multi-GPU training instead of DistributedDataParallel. If not given, DistributedDataParallel is used.'
                        )
    parser.add_argument(
        '-ddp', '--distributed', action='store_true', required=False, help='Use DistributedDataParallel even for single GPU training.'
                        ) 
    parser.add_argument(
        '-tf', '--train_folder', type=str, required=False,
        help='Destination of training images. If not given, random data is used.'
                        )
    parser.add_argument(
        '-vf', '--val_folder', type=str, required=False,
        help='Destination of validation images. If not given, random data is used.'
                        )
    parser.add_argument(
        '-nw', '--num_workers', type=int, required=False,
        help='Number of workers for the dataloader. If not given 2*num_gpus is used.'
                        )
    parser.add_argument(
        '-sd', '--split_data', type=float, default=1, required=False,
        help='Splits dataset in training and evaluation dataset with given ratio.'
                        )
    parser.add_argument(
        '-le', '--load_from_epoch', type=int, default=0, required=False,
        help='Loads model state at given epoch. '
             'If -1 is given, the highest available epoch for the model and the dataset is used.'
                        )
    parser.add_argument(
        '-nt', '--no_temp', action='store_true', required=False,
        help='Hide GPU infos like temperature etc. for better performance'
                        )
    parser.add_argument(
        '-ev', '--eval', action='store_true', required=False,
        help='If given, Evaluates the model with the Validation dataset '
             'at the end of each epoch and prints the evaluation accuracy.'
                        )
    parser.add_argument(
        '-eo', '--eval_only', action='store_true', required=False,
        help='Evaluation mode with metaparameters optimized for evaluation. '
             'If not given, training mode with metaparameters optimized for training is used'
                        )
    parser.add_argument(
        '-mi', '--mean_img_per_sec', action='store_true', required=False,
        help='Plot mean images per second at the end and save it in the logfile'
                        )
    parser.add_argument(
        '-s', '--set_seed', type=int, default=1234, required=False,
        help='Set the random seed used for splitting dataset into train and eval sets. Default: 1234'
                        )
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=1e-3, required=False,
        help='Set the learning rate for training. Default: 1e-3'
                        )
    parser.add_argument(
        '-slr', '--step_lr', type=int, default=30, required=False,
        help='Decay the learning rate after by factor 10 every given epoch.'
                        )
    parser.add_argument(
        '-ce', '--calc_every', type=int, default=10, required=False,
        help='Set the stepsize for calculations and print outputs. Default: 10'
                        )
    parser.add_argument(
        '-ri', '--refresh_interval', type=int, default=500, required=False,
        help='Change live plot refresh interval in ms.'
                        )
    parser.add_argument(
        '-lp', '--live_plot', action='store_true', required=False,
        help='Show live plot of gpu temperature and fan speed.'
                        )
    parser.add_argument(
        '-pl', '--pred_pic_label', type=str, required=False,
        help='Predict label of given picture with pretrained model.'
                        )
    parser.add_argument(
        '-in', '--imagenet', type=str, required=False,
        help='Use imagenet for training/evaluation from given path.'
                        )
    parser.add_argument(
        '-ln', '--log_file', required=False, nargs='?', const='', type=str,
        help='Make a logfile and save it in /log/ under given name. If no name is given,'
             '<num_gpus>_<GPU_name>_<model>_<batch_size>_<learning_rate> is used' 
                        )
    parser.add_argument(
        '-cf', '--checkpoint_folder', type=str, required=False,
        help='Save training checkpoints in given folder name.'
                        )
    parser.add_argument(
        '-op', '--optimizer', type=str, default='SGD', required=False,
        help='Set the optimizer. Default: SGD.'
                        )
    parser.add_argument(
        '-do', '--dist_optim', action='store_true', required=False,
        help='Use distributed optimizer (ZeroRedundancyOptimizer). (Experimental)'
                        )
    parser.add_argument(
        '-do9', '--dist_optim_190', action='store_true', required=False,
        help='Use distributed Optimizer (ZeroRedundancyOptimizer) from torch.distributed.optim '
             '(available in Pytorch 1.9.0., experimental).'
                        )
    parser.add_argument(
        '-dm', '--distribution_mode', type=int, required=False,
        help='Set distribution mode: 0 : None, 1: DistributedDataParallel (same as --distributed), '
             '2 : DataParallel (same as --parallel)'
                        )
    parser.add_argument(
        '-pm', '--pin_memory', type=bool, default=True, required=False,
        help='Use pin_memory = True in the dataloader and set the output labels to cuda(NonBlocking=True)'
                        )
    parser.add_argument(
        '-bb', '--benchmark_backend', type=bool, default=True, required=False,
        help='Use torch.backends.cudnn.benchmark = False.'
                        )
    parser.add_argument(
        '-ni', '--num_images', type=int, default=100000, required=False,
        help='Number of images in the random image dataset.'
                        )
    parser.add_argument(
        '-ad', '--average_duration', action='store_true', required=False,
        help='Calculate the average of the durations measured by each gpu.'
             'The duration is needed to get the images per second.'
                        )
    parser.add_argument(
        '-ag', '--average_gradients', action='store_true', required=False,
        help='Average the gradients of the model after each step on the cost of performance.'
                        )
    parser.add_argument(
        '-pb', '--process_group_backend', type=str, default='nccl', required=False,
        help='Choose a different backend for the distribution process group.'
             '"nccl" is supposed to have more features for distributed GPU training, '
             'but the performance on resnet50 seems to be better with "gloo". '
             'Default: "gloo"'
                        )
    parser.add_argument(
        '-lb', '--log_benchmark', action='store_true', required=False,
        help='Write all the benchmark results into the log file.'
                        )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.benchmark_backend

    if not args.num_workers:
        args.num_workers = 2 * args.num_gpus

    if args.log_file is None:
        args.log_file = f'{args.num_gpus}_{str(torch.cuda.get_device_name(0)).replace(" ","").replace("/","")}'\
                        f'_{args.model}_{args.batch_size}_lr{str(args.learning_rate).replace(".", "")}.txt'

    if (args.eval and not (args.split_data != 1 or args.val_folder)) and not args.imagenet:
        args.split_data = 0.9

    if args.use_fp16:
        args.precision = 'half'
    else:
        args.precision = 'float'

    if not args.distributed:
        args.distributed = not args.parallel and args.num_gpus > 1
        
    if args.distributed:
        args.batch_size = int(args.batch_size / args.num_gpus)

    if args.checkpoint_folder:
        args.checkpoint_folder = Path(__file__).absolute().parent.parent / 'model_checkpoints' / args.checkpoint_folder
    else:
        args.checkpoint_folder = Path(__file__).absolute().parent.parent / 'model_checkpoints' / args.log_file.replace('.txt', '').replace('.log', '')  #ab Python 3.9: .removesuffix('.txt').removesuffix('.log')

    if args.eval_only:
        args.num_epochs = 1
        if args.load_from_epoch == 0:  
            data_name = MultiGpuData.make_data_name(args)
            sys.exit(
                f'Evaluation with untrained model not possible. '
                f'Load a pretrained model with "--load_from_epoch <epoch>"\n'
                f'Highest available epoch for {args.model} with {data_name}: '
                f'{MultiGpuModel.check_saved_checkpoint_epoch(args.model, data_name, args.checkpoint_folder)}'
                     )

    if not 0 < args.split_data <= 1:
        sys.exit('--split_data has to be between 0 and 1')


    if args.distribution_mode is None:
        args.distribution_mode = int(args.distributed) + 2 * int(args.parallel)

    if args.load_from_epoch == -1:
        data_name = MultiGpuData.make_data_name(args)
        args.load_from_epoch = MultiGpuModel.check_saved_checkpoint_epoch(args.model, data_name, args.checkpoint_folder)

    return args

def string_or_none(str):
    if not str:
        return None
    return str
