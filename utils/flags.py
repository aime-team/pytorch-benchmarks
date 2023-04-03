import argparse
import sys
import os
import torch
from pathlib import Path
from data.data import MultiGpuData
from models.utils import MultiGpuModel



class Flags():

    def __init__(self):
        self.args = self.load_flags()
        self.set_batch_size()
        self.init_model()
        self.set_default_values()
        self.set_precision_mode()
        self.set_distribution_mode()
        self.check_eval_only_mode()
        self.init_benchmark_mode()

    def load_flags(self):
        """Parses arguments and sets global parameters.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        parser = argparse.ArgumentParser(
            description='PyTorch Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                        )

        parser.add_argument(
            '-ne', '--num_epochs', type=int, default=10, required=False, help='Number of epochs. Default: 10'
                            )
        parser.add_argument(
            '-b', '--batch_size', type=int, required=False, help='Local batch size for training. Default: 64'
                            )
        parser.add_argument(
            '-gb', '--global_batch_size', type=int, required=False, help='Global batch size for training. Default: 64 * num_gpus'
                            )
        parser.add_argument(
            '-eb', '--eval_batch_size', type=int, help='Local batch size for evaluation. Default: batch size for training: 64'
                            )
        parser.add_argument(
            '-geb', "--global_eval_batch_size", type=int, help='Global batch size for evaluation. Default: global batch size for training: 64 * num_gpus'
                            )
        parser.add_argument(
            '-ng', '--num_gpus', type=int, default=1, required=False, help='Number of gpus used for training. Default: 1'
                            )
        parser.add_argument(
            '-m', '--model', type=str, default='resnet50', required=False,
            help='Choose the model. Default: resnet50'
                            )
        parser.add_argument(
            '-pt2', '--compile', action='store_true', required=False,
            help='Do optimizations for Pytorch 2. Does not work with Pytorch 1. Default: False'
                            )
        parser.add_argument(
            '-ev', '--eval', action='store_true', required=False,
            help='If given, the model is evaluated with the given validation dataset in --eval_image_folder or --eval_data_file'
                'at the end of each epoch and prints the evaluation accuracy.'
                            )
        parser.add_argument(
            '-eo', '--eval_only', action='store_true', required=False,
            help='If given, the model is evaluated with the given validation dataset without any training.'
                            )
        parser.add_argument(
            '-nw', '--num_workers', type=int, required=False,
            help='Number of workers for the dataloader. If not given 4*num_gpus is used.'
                            )
        parser.add_argument(
            '-amp', '--auto_mixed_precision', action='store_true', required=False,
            help='Enable automatic mixed precision. Default: False'
                            )
        parser.add_argument(
            '-dp', '--parallel', action='store_true', required=False, help='Use DataParallel for Multi-GPU training instead of DistributedDataParallel. If not given, DistributedDataParallel is used.'
                            )
        parser.add_argument(
            '-ddp', '--distributed', action='store_true', required=False, help='Use DistributedDataParallel even for single GPU training.'
                            ) 
        parser.add_argument(
            '-tf', '--train_image_folder', type=str, required=False,
            help='Destination of the training image dataset. If not given, synthetic data is used.'
                            )
        parser.add_argument(
            '-ef', '--eval_image_folder', type=str, required=False,
            help='Destination of the validation image dataset. If not given, synthetic data is used.'
                            )
        parser.add_argument(
            '-lr', '--learning_rate', type=float, default=1e-3, required=False,
            help='Set the learning rate for training. Default: 1e-3'
                            )
        parser.add_argument(
            '-slr', '--step_lr', type=int, default=30, required=False,
            help='Decay the learning rate by factor 10 every given epoch. Default: 30'
                            )
        parser.add_argument(
            '-ldf', '--lr_decay_factor', type=float, default=10, required=False,
            help='Change the factor of the learning rate decay. Default: 10'
                            )
        parser.add_argument(
            '-cl', '--constant_learning_rate', action='store_true', required=False,
            help='Train with a constant learning rate'
                            )
        parser.add_argument(
            '-sd', '--split_data', type=float, default=1, required=False,
            help='Splits the given training dataset in a training and evaluation dataset with the given ratio. Takes values between 0 and 1. Default: 1'
                            )
        parser.add_argument(
            '-le', '--load_from_epoch', type=int, default=0, required=False,
            help='Loads model state at given epoch of a saved checkpoint given in --checkpoint_folder. '
                'If -1 is given, the highest available epoch for the model and the dataset is used.'
                            )
        parser.add_argument(
            '-w',  '--warm_up_steps', type=int, default=10, required=False,
            help="Number of warm up steps in every epoch. Warm up steps will not taken into account. Default: 10"
                            )
        parser.add_argument(
            '-nt', '--no_temp', action='store_true', required=False,
            help='Hide GPU infos like temperature etc. for better performance'
                            )
        parser.add_argument(
            '-f', '--use_fp16', action='store_true', required=False, help='Use half precision. If not given fp32 precision is used.'
                            )
        parser.add_argument(
            '-mi', '--mean_it_per_sec', action='store_true', required=False,
            help='Plot mean images/sequences per second at the end and save it in the logfile'
                            )
        parser.add_argument(
            '-ce', '--calc_every', type=int, default=10, required=False,
            help='Set the stepsize for calculations and print outputs. Default: 10'
                            )
        parser.add_argument(
            '-lp', '--live_plot', action='store_true', required=False,
            help='Show live plot of gpu temperature and fan speed.'
                            )
        parser.add_argument(
            '-ri', '--refresh_interval', type=int, default=500, required=False,
            help='Change live plot refresh interval in ms.'
                            )
        parser.add_argument(
            '-pl', '--pred_pic_label', type=str, required=False,
            help='Predict label of given picture with a pretrained model given in --checkpoint_folder.'
                            )
        parser.add_argument(
            '-in', '--imagenet', type=str, required=False,
            help='Use imagenet for training/evaluation from given path.'
                            )
        parser.add_argument(
            '-ln', '--log_file', required=False, nargs='?', const='', type=str,
            help='Save the logfile in /log/ under given name. If no name is given,'
                '<num_gpus>_<GPU_name>_<model>_<batch_size>_<learning_rate> is used' 
                            )
        parser.add_argument(
            '-cf', '--checkpoint_folder', type=str, required=False,
            help='Save training checkpoints in given folder name. If not given, the name of the log-file is used.'
                            )
        parser.add_argument(
            '-cfi', '--checkpoint_file', type=str, required=False,
            help='Load checkpoint from given file.'
                            )
        parser.add_argument(
            '-op', '--optimizer', type=str, required=False,
            help='Set the optimizer. Default: Adam for BERT, SGD for resnet50.'
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
            '-bb', '--benchmark_backend', action='store_true', required=False,
            help='Use torch.backends.cudnn.benchmark = True.'
                            )
        parser.add_argument(
            '-syn', '--synthetic_data', action='store_true', required=False,
            help='Use synthetic data. Default: False (True, if no data is given)'
                            )
        parser.add_argument(
            '-ni', '--num_synth_data', type=int, default=100000, required=False,
            help='Number of examples in the synthetic dataset.'
                            )
        parser.add_argument(
            '-ad', '--average_duration', action='store_true', required=False,
            help='Calculate the average of the durations measured by each gpu.'
                'The duration is needed to get the images per second.'
                            )
        parser.add_argument(
            '-ag', '--average_gradients', action='store_true', required=False,
            help='Average the gradients of the model after each step on the cost of performance (Experimental, no improvement in training).'
                            )
        parser.add_argument(
            '-pb', '--process_group_backend', type=str, default='nccl', required=False,
            help='Choose a different backend for the distribution process group (nccl or gloo). "nccl" is supposed to have more features for distributed GPU training. Default: "nccl"'
                            )
        parser.add_argument(
            '-lb', '--log_benchmark', action='store_true', required=False,
            help='Write all the benchmark results into the log file.'
                            )
        parser.add_argument(
            '-na', '--no_augmentation', action='store_true', required=False,
            help='No augmentation of the image training dataset.'
                            )
        parser.add_argument(
            '-pt', '--pretrained', action='store_true', required=False,
            help='Load pretrained model from --checkpoint_file. Default: False.' 
                            )
        parser.add_argument(
            '-bt', '--benchmark_train', action='store_true', required=False,
            help='Start training for benchmark. Default: False'
                            )
        parser.add_argument(
            '-bts', '--benchmark_train_steps', type=int, default=60, required=False,
            help='Number of steps for --benchmark_train.'
                            )
        parser.add_argument(
            '-dn', '--data_name', type=str, required=False, 
            help='Own name of dataset used for training/evaluation, if not ImageNet or Squad is used.'
                            )
        parser.add_argument(
            '-tdf', '--train_data_file', type=str, required=False, 
            help='For BERT: SQuAD json for training. E.g., train_squad.json'
                            )
        parser.add_argument(
            '-edf', '--eval_data_file', type=str, required=False, 
            help='For BERT: SQuAD json for evluation. E.g., eval_squad.json'
                            )
        parser.add_argument(
            '-msl', '--max_seq_length', default=384, type=int, required=False, 
            help='For BERT: The maximum total input sequence length after WordPiece tokenization. Sequences '
            'longer than this will be truncated, and sequences shorter than this will be padded.'
                            )
        parser.add_argument(
            '-ds', '--doc_stride', default=128, type=int, required=False, 
            help='For BERT: When splitting up a long document into chunks, how much stride to take between chunks.'
                            )
        parser.add_argument(
            '-mql', '--max_query_length', default=64, type=int, required=False, 
            help='For BERT: The maximum number of tokens for the question. Questions longer than this will '
            'be truncated to this length.'
                            )
        parser.add_argument(
            '-wp', '--warmup_proportion', default=0.1, type=float, required=False, 
            help='For BERT: Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% '
            'of training.'
                            )
        parser.add_argument(
            '-nbs', '--n_best_size', default=20, type=int, required=False, 
            help='For BERT: The total number of n-best predictions.'
                            )
        parser.add_argument(
            '-mal', '--max_answer_length', default=30, type=int, required=False, 
            help='For BERT: The maximum length of an answer that can be generated. This is needed because the start '
            'and end predictions are not conditioned on one another.'
                            )
        parser.add_argument(
            '-gas', '--gradient_accumulation_steps', type=int, default=1, required=False, 
            help='Number of updates steps to accumulate before performing a backward/update pass.'
                            )
        parser.add_argument(
            '-dlc', '--do_lower_case', action='store_true', required=False, 
            help='For BERT: Whether to lower case the input text. True for uncased models, False for cased models.'
                            )
        parser.add_argument(
            '-v2n', '--version_2_with_negative', action='store_true', required=False, 
            help='If true, the SQuAD examples contain some that do not have an answer.'
                            )
        parser.add_argument(
            '-nsd', '--null_score_diff_threshold', type=float, default=0.0, required=False, 
            help='If null_score - best_non_null is greater than the threshold predict null.'
                            )
        parser.add_argument(
            '-vf', '--vocab_file', type=str, required=False, 
            help='For BERT: Vocabulary mapping/file BERT was pretrainined on.'
                            )
        parser.add_argument(
            '-sc', '--skip_checkpoint', default=False, action='store_true', required=False, 
            help='Whether to save checkpoints'
                            )
        parser.add_argument(
            '-sca', '--skip_cache', default=False, action='store_true', required=False, 
            help='For BERT: Whether to cache train/evaluation features'
                            )
        parser.add_argument(
            '-ctf', '--cached_train_features_file', type=str, required=False, 
            help='For BERT: Location to cache train feaures. Will default to the dataset directory'
                            )
        parser.add_argument(
            '-cef', '--cached_eval_features_file', type=str, required=False, 
            help='For BERT: Location to cache evaluation feaures. Will default to the dataset directory'
                            )
        parser.add_argument(
            '-rnc', '--renew_cache', action='store_true', required=False, 
            help='For BERT: Rebuild cache file of training/evaluation data.'
                            )   
        parser.add_argument(
            '-se', '--seed', type=int, required=False, 
            help='Random seed for initialization.'
                            )
        parser.add_argument(
            '-nb', '--no_benchmark', action='store_true', required=False,
            help="Don't do benchmark calculations"
                            )   
        parser.add_argument(
            '-up', '--find_unused_parameters', action='store_true', required=False, default=False,
            help='Enabling unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`'
                            )
        parser.add_argument(
            '-gmb', '--get_max_batchsize', action='store_true', required=False, default=False,
            help='Getting the maximum possible batch_size for given model'
                            )
                        
        

        """                        
        parser.add_argument(
            '-bv', '--benchmark_val', action='store_true', required=False,
            help='Start validation for benchmark. Default: False'
                            )
        """

        args = parser.parse_args()
        return args

    def set_batch_size(self):

        if not self.args.batch_size:
            if self.args.global_batch_size:
                self.args.batch_size = int(self.args.global_batch_size / self.args.num_gpus) 
            else:
                self.args.batch_size = 64
                self.args.global_batch_size = self.args.batch_size * self.args.num_gpus
        else:
            if self.args.global_batch_size and self.args.global_batch_size != self.args.num_gpus * self.args.batch_size:
                sys.exit("global_batch_size is supposed to be local_batch_size * num_gpus, if they are both given. Choose either or.")
            else:
                self.args.global_batch_size = self.args.batch_size * self.args.num_gpus
        if not self.args.eval_batch_size:
            if self.args.global_eval_batch_size:
                self.args.eval_batch_size = int(self.args.global_eval_batch_size / self.args.num_gpus) 
            else:
                self.args.eval_batch_size = self.args.batch_size
                self.args.global_eval_batch_size = self.args.eval_batch_size * self.args.num_gpus
        else:
            if self.args.global_eval_batch_size and self.args.global_eval_batch_size != self.args.num_gpus * self.args.eval_batch_size:
                sys.exit("global_eval_batch_size is supposed to be eval_batch_size * num_gpus, if they are both given. Choose either or.")
            else:
                self.args.global_eval_batch_size = self.args.batch_size * self.args.num_gpus



    def init_model(self):
        if self.args.model == 'bert-base-uncased':
            self.args.bert = True
            self.args.do_lower_case = True
            self.args.bert_config_dict = {
                "attention_probs_dropout_prob": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "max_position_embeddings": 512,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "type_vocab_size": 2,
                "vocab_size": 30522
                        }
            if not self.args.vocab_file:
                self.args.vocab_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_base' / 'bert_base_uncased_vocab.txt'
            if not self.args.checkpoint_file:
                self.args.checkpoint_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_base' / 'bert_base_uncased.pt'

        elif self.args.model == 'bert-large-uncased':
            self.args.bert = True
            self.args.do_lower_case = True
            self.args.bert_config_dict = {
                "attention_probs_dropout_prob": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "max_position_embeddings": 512,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "type_vocab_size": 2,
                "vocab_size": 30522
                        }
            if not self.args.vocab_file:
                self.args.vocab_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_large' / 'bert_large_uncased_vocab.txt'
            if not self.args.checkpoint_file:
                self.args.checkpoint_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_large' / 'bert_large_uncased.pt'

        elif self.args.model == 'bert-base-cased':
            self.args.bert = True
            self.args.do_lower_case = False
            self.args.bert_config_dict = {
                "attention_probs_dropout_prob": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "max_position_embeddings": 512,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "type_vocab_size": 2,
                "vocab_size": 28996
                                    }
            if not self.args.vocab_file:
                self.args.vocab_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_base' / 'bert_base_cased_vocab.txt'
            if not self.args.checkpoint_file:
                self.args.checkpoint_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_base' / 'bert_base_cased.pt'

        elif self.args.model == 'bert-large-cased':
            self.args.bert = True
            self.args.do_lower_case = False
            self.args.bert_config_dict = {
                "attention_probs_dropout_prob": 0.1, 
                "directionality": "bidi", 
                "hidden_act": "gelu", 
                "hidden_dropout_prob": 0.1, 
                "hidden_size": 1024, 
                "initializer_range": 0.02, 
                "intermediate_size": 4096, 
                "max_position_embeddings": 512, 
                "num_attention_heads": 16, 
                "num_hidden_layers": 24, 
                "pooler_fc_size": 768, 
                "pooler_num_attention_heads": 12, 
                "pooler_num_fc_layers": 3, 
                "pooler_size_per_head": 128, 
                "pooler_type": "first_token_transform", 
                "type_vocab_size": 2, 
                "vocab_size": 28996
                                    }
            if not self.args.vocab_file:
                self.args.vocab_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_large' / 'bert_large_cased_vocab.txt'
            if not self.args.checkpoint_file:
                self.args.checkpoint_file = Path(__file__).absolute().parent.parent / 'data'/ 'bert_large' / 'bert_large_cased.pt'
        else:
            self.args.bert = False

    def set_default_values(self):
        if not self.args.optimizer:
            if self.args.bert:
                self.args.optimizer = 'Adam'
            else:
                self.args.optimizer = 'SGD'
        self.args.num_train_optimization_steps = -1
        if self.args.bert:
            self.args.iterations = 'Sequences'
            self.args.find_unused_parameters = True
        else:
            self.args.iterations = 'Images' 
        if self.args.num_workers is None:
            self.args.num_workers = 4 * self.args.num_gpus
        cpu_threads = os.cpu_count()
        if self.args.num_workers >= cpu_threads:
            self.args.num_workers = cpu_threads
            print(f'Too much workers for the dataloader! Your cpu has only {cpu_threads} threads, so the number of workers was set to {cpu_threads}')

        if self.args.log_file is None:
            self.args.log_file = f'{self.args.num_gpus}_{str(torch.cuda.get_device_name(0)).replace(" ","").replace("/","")}'\
                            f'_{self.args.model}_{self.args.batch_size}_lr{str(self.args.learning_rate).replace(".", "")}.txt'

        if self.args.eval_image_folder:
            self.args.eval = True

        if (self.args.eval and not (self.args.split_data != 1 or self.args.eval_image_folder)) and not self.args.imagenet:
            self.args.split_data = 0.9
        if self.args.load_from_epoch == -1:
            self.args.load_from_epoch = MultiGpuModel.check_saved_checkpoint_epoch(self.args.model, self.args.data_name, self.args.checkpoint_folder)
        if self.args.checkpoint_folder:
            self.args.checkpoint_folder = Path(__file__).absolute().parent.parent / 'model_checkpoints' / self.args.checkpoint_folder
        else:
            self.args.checkpoint_folder = Path(__file__).absolute().parent.parent / 'model_checkpoints' / self.args.log_file.replace('.txt', '').replace('.log', '')  #ab Python 3.9: .removesuffix('.txt').removesuffix('.log')

        if not self.args.checkpoint_file:
            self.args.checkpoint_file = self.args.checkpoint_folder / f'{self.args.model}_{self.args.data_name}_epoch_{self.args.load_from_epoch}.pt'
        if self.args.constant_learning_rate:
            self.args.lr_decay_factor = 1

        if not 0 < self.args.split_data <= 1:
            sys.exit('--split_data has to be between 0 and 1')



        if self.args.imagenet:
            self.args.data_name  = 'ImageNet'
        elif self.args.train_image_folder:
            data_name_list = self.args.train_image_folder.strip('/').split('/')
            self.args.data_name  = f'{data_name_list[-2]}_{data_name_list[-1]}'

        elif self.args.data_name and self.args.data_name.lower() == 'squad':
            self.args.data_name = 'SQuAD'
            if not self.args.train_data_file:
                self.args.train_data_file = Path(__file__).absolute().parent.parent / 'data' / 'squad' / 'train_squad.json'

            if not self.args.eval_data_file:
                self.args.eval_data_file = Path(__file__).absolute().parent.parent / 'data' / 'squad' / 'eval_squad.json'
        
        elif not self.args.train_data_file:
            self.args.synthetic_data = True
            self.args.data_name = 'Synthetic data'
    




        if self.args.bert and self.args.train_data_file:        
            if self.args.cached_train_features_file is None:
                self.args.cached_train_features_file = self.args.train_data_file.parent / 'cache' / f'{self.args.train_data_file.stem}_{self.args.model}_{self.args.max_seq_length}_{self.args.doc_stride}_{self.args.max_query_length}.json'
                if not self.args.cached_train_features_file.parent.is_dir():
                    self.args.cached_train_features_file.parent.mkdir() 
            
            if self.args.cached_eval_features_file is None:
                self.args.cached_eval_features_file = self.args.eval_data_file.parent / 'cache' / f'{self.args.eval_data_file.stem}_{self.args.model}_{self.args.max_seq_length}_{self.args.doc_stride}_{self.args.max_query_length}.json'
                if not self.args.cached_eval_features_file.parent.is_dir():
                    self.args.cached_eval_features_file.parent.mkdir() 

        if self.args.gradient_accumulation_steps < 1:
            raise ValueError(f'Invalid gradient_accumulation_steps parameter: {self.args.gradient_accumulation_steps}, should be >= 1')
        else:
            self.args.batch_size = self.args.batch_size // self.args.gradient_accumulation_steps

    def set_precision_mode(self):

        if self.args.use_fp16:
            self.args.precision = 'half'
        else:
            self.args.precision = 'float'

    def set_distribution_mode(self):

        if not self.args.distributed:
            self.args.distributed = not self.args.parallel and self.args.num_gpus > 1

        if self.args.distribution_mode == 2:
            self.args.parallel = True
            self.args.distributed = False
        if self.args.distribution_mode == 1:
            self.args.distributed = True
        if self.args.distribution_mode == 0 and self.args.num_gpus > 1:
            sys.exit('Choose either --num_gpus 1 or --distribution_mode 1 or 2.')
        elif self.args.distribution_mode is None:
            self.args.distribution_mode = int(self.args.distributed) + 2 * int(self.args.parallel)


    def check_eval_only_mode(self):

        if self.args.eval_only:
            self.args.eval = True
            self.args.num_epochs = 1
            if self.args.pretrained or self.args.synthetic_data:
                pass           
            else:
                if self.args.load_from_epoch == 0:  
                    sys.exit(
                        f'Evaluation with untrained model not possible. '
                        f'Load a pretrained model with "--load_from_epoch <epoch>"\n'
                        f'Highest available epoch for {self.args.model} with {self.args.data_name}: '
                        f'{MultiGpuModel.check_saved_checkpoint_epoch(self.args.model, self.args.data_name, self.args.checkpoint_folder)}'
                            )
            

    def init_benchmark_mode(self):

        if self.args.benchmark_train:
            self.args.synthetic_data = True
            self.args.num_epochs = 1
            self.args.mean_it_per_sec = True
            self.args.num_synth_data = self.args.benchmark_train_steps*self.args.batch_size*self.args.num_gpus
        """        
        if self.args.benchmark_val:
            self.args.num_epochs = 1
            self.args.num_synth_data = 5000
            self.args.mean_it_per_sec = True
            self.args.eval_only = True
        """
        torch.backends.cudnn.benchmark = self.args.benchmark_backend


def string_or_none(str):
    if not str:
        return None
    return str
