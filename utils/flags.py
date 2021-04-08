import argparse
import models.model_utils as model_utils


def load_flags():
    """Parses arguments.
    """

    parser = argparse.ArgumentParser(description='PyTorch Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-w',  '--warm_up_steps', type=int, default=10, required=False,
        help="Number of warm up steps in every epoch. Warm up steps will not taken into account"
                        )
    parser.add_argument(
        '-n', '--num_epochs', type=int, default=10, required=False, help='Number of total epochs. '
                        )
    parser.add_argument(
        '-b', '--batch_size', type=int, default=64, required=False, help='Batch size. Default: 64'
                        )
    parser.add_argument(
        '-g', '--num_gpus', type=int, default=1, required=False, help='Number of gpus used for training'
                        )
    parser.add_argument(
        '-i', '--gpu_ids', type=int, nargs='+', required=False,
        help='IDs of used GPUs for training. If not given, range(num_gpus) is used.'
                        )
    parser.add_argument(
        '-m', '--model', type=str, default='resnet50', required=False,
        help='Use different model from torchvision. Default: resnet50'
                        )
    parser.add_argument(
        '-f', '--use_fp16', action='store_true', required=False, help='Use half precision'
                        )
    parser.add_argument(
        '-dp', '--parallel', action='store_true', required=False, help='Use DataParallel instead of Distributed Data'
                        )
    parser.add_argument(
        '-imf', '--img_folder', type=str, required=False,
        help='Destination of training images. If not given, random data is used.'
                        )
    parser.add_argument(
        '-nw', '--num_workers', type=int, default=-1, required=False,
        help='Number of workers for the dataloader. If not given num_gpus is used.'
                        )
    parser.add_argument(
        '-sd', '--split_data', type=float, default=1, required=False,
        help='Splits dataset in training and evaluation dataset with given ratio.'
                        )
    parser.add_argument(
        '-se', '--start_epoch', type=int, default=1, required=False,
        help='Resumes training of saved model at given epoch. If 0 is given, the highest available epoch is used.'
                        )
    parser.add_argument(
        '-nt', '--no_temp', action='store_true', required=False,
        help='Hide GPU infos like temperature etc. for better performance'
                        )
    parser.add_argument(
        '-ev', '--eval', action='store_true', required=False,
        help='Evaluation mode with metaparameters optimized for evaluation. '
             'If not given, training mode with metaparameters optimized for training is used'
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


    args = parser.parse_args()
    if args.eval and args.start_epoch == 1:
        sys.exit(
            f'Evaluation with untrained model not possible. Load a pretrained model with "--load_model <epoch>"\n'
            f'Highest available epoch for {args.model}: '
            f'{model_utils.MultiGpuModel.check_saved_checkpoint_epoch(args.model)}'
                 )
    if args.use_fp16:
        args.precision = 'half'
    else:
        args.precision = 'float'
    if args.parallel:
        args.batch_size *= args.num_gpus
    if not 0 < args.split_data <= 1:
        sys.exit('--split_data has to be between 0 and 1')
    return args

