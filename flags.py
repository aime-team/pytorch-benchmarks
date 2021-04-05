import argparse
import model_utils


def load_flags():
    """Parses arguments.
    """

    parser = argparse.ArgumentParser(description='PyTorch Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-w', '--warm_up_steps', type=int, default=10, required=False,
        help="Number of warm up steps in every epoch. Warm up steps will not taken into account"
                        )
    parser.add_argument(
        '--num_epochs', '-n', type=int, default=10, required=False, help='Number of total epochs. '
                        )
    parser.add_argument(
        '--batch_size', '-b', type=int, default=48, required=False, help='Batch size. Default: 64'
                        )
    parser.add_argument(
        '--num_gpus', '-g', type=int, default=1, required=False, help='Number of gpus used for training'
                        )
    parser.add_argument(
        '--gpu_ids', '-i', type=int, nargs='+', required=False,
        help='IDs of used GPUs for training. If not given, range(num_gpus) is used.'
                        )
    parser.add_argument(
        '--model', '-m', type=str, default='resnet50', required=False,
        help='Use different model from torchvision. Default: resnet50'
                        )
    parser.add_argument(
        '--use_fp16', '-f', action='store_true', required=False, help='Use half precision'
                        )
    parser.add_argument(
        '--parallel', '-dp', action='store_true', required=False, help='Use DataParallel instead of Distributed Data'
                        )
    parser.add_argument(
        '--img_folder', '-imf', type=str, required=False,
        help='Destination of training images. If not given, random data is used.'
                        )
    parser.add_argument(
        '--num_workers', '-nw', type=int, default=-1, required=False,
        help='Number of workers for the dataloader. If not given num_gpus is used.'
                        )
    parser.add_argument(
        '--split_data', '-sd', type=float, default=-1, required=False,
        help='Splits dataset in training and evaluation dataset with given ratio.'
                        )
    parser.add_argument(
        '--start_epoch', '-se', type=int, default=1, required=False,
        help='Resumes training of saved model at given epoch. If 0 is given, the highest available epoch is used.'
                        )
    parser.add_argument(
        '--no_temp', '-st', action='store_true', required=False,
        help='Hide GPU infos like temperature etc. for better performance'
                        )
    parser.add_argument(
        '--eval', '-ev', action='store_true', required=False,
        help='Evaluation mode with metaparameters optimized for evaluation. '
             'If not given, training mode with metaparameters optimized for training is used'
                        )
    parser.add_argument(
        '--set_seed', '-s', type=int, default=1234, required=False,
        help='Set the random seed used for splitting dataset into train and eval sets. Default: 1234'
                        )
    parser.add_argument(
        '--learning_rate', '-lr', type=float, default=1e-3, required=False,
        help='Set the learning rate for training. Default: 1e-3'
                        )
    parser.add_argument(
        '--calc_every', '-ce', type=int, default=10, required=False,
        help='Set the stepsize for calculations and print outputs. Default: 10'
                        )
    parser.add_argument(
        '-ri', '--refresh_interval', type=int, default=500, required=False,
        help='Change live plot refresh interval in ms.'
                        )
    parser.add_argument(
        '--live_plot', '-nl', action='store_true', required=False,
        help='No live plot of gpu temperature and fan speed for better performance.'
                        )

    args = parser.parse_args()
    if args.eval and args.start_epoch == 1:
        sys.exit(
            f'Evaluation with untrained model not possible. Load a pretrained model with "--load_model <epoch>"\n'
            f'Highest available epoch for {args.model}: '
            f'{model_utils.MultiGpuModel.check_pretrained_model_epoch(args.model)}'
                 )
    if args.use_fp16:
        args.precision = 'half'
    else:
        args.precision = 'float'
    if args.parallel:
        args.batch_size *= args.num_gpus
    return args

