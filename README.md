# PyTorch-Benchmarks

Tool to benchmark multi-GPU systems and to optimize multi-GPU training with PyTorch.  
  
## Usage
Run main.py to start benchmarking!

## Optional arguments:
  -w, --warm_up_steps : Number of warm up steps in every epoch. Warm up steps will not taken into account; default: 10  
  -n, --num_epochs: Number of epochs; default: 10  
  -b, --batch_size : Batch size; default: 64  
  -ng, --num_gpus :Number of gpus used for training; default: 1  
  -m, --model : Model used for training; default: resnet50  
  -f, --use_fp16 : Use half precision; default: False  
  -imf, --img_folder : Destination of training images. If not given, random data is used.  
  -dp, --parallel: Use DataParallel for Multi-GPU training instead of DistributedDataParallel. If not given, DistributedDataParallel is used.  
  -ddp, --distributed: Use DistributedDataParallel even for single GPU training.  
  -tf, --train_folder: Destination of training images. If not given, random data is used.  
  -vf, --val_folder, Destination of training images. If not given, random data is used.  
  -nw, --num_workers : Number of workers used by the dataloader. If not given, num_gpus is used.  
  -sd, --split_data: Splits dataset in training and evaluation dataset with given ratio.  
  -le, --load_from_epoch: Loads model state at given epoch. If -1 is given, the highest available epoch for the model and the dataset is used.  
  -nt, --no_temp: Hide GPU infos like temperature etc. for better performance.  
  -ev, --eval: If given, evaluation of the model with the validation dataset will be calculated at the end of each epoch and the evaluation accuracy will be printed.  
  -eo, --eval_only: Evaluation mode with metaparameters optimized for evaluation. If not given, training mode with metaparameters optimized for training is used.  
  -mi, --mean_img_per_sec: Plot mean images per second at the end and save it in the logfile.  
  -s, --set_seed: Set the random seed used for splitting dataset into train and eval sets. Default: 1234  
  -lr, --learning_rate: Set the learning rate for training. Default: 1e-3  
  -ce, --calc_every: Set the stepsize for calculations and print outputs. Default: 10  
  -ri, --refresh_interval: Change live plot refresh interval in ms. Default: 500  
  -lp, --live_plot: Show live plot of gpu temperature and fan speed.  
  -pl, --pred_pic_label: Predict label of given picture with pretrained model.  
  -in, --imagenet: Use imagenet for training/evaluation from given path.  
  -ln, --log_file: Make a logfile and save it in /log/ under given name. If no name is given, '<num_gpus>_<GPU_name>_<batch_size>' is used.  
  -cf, --checkpoint_folder: Save training checkpoints in given folder name.  
  -do, --dist_optim: Use distributed optimizer (ZeroRedundancyOptimizer). (Experimental)  
  -do9, --dist_optim_190: Use distributed Optimizer (ZeroRedundancyOptimizer) from torch.distributed.optim (available in Pytorch 1.9.0., experimental).  
  -dm, --distribution_mode: Set distribution mode: 0 : None, 1: DistributedDataParallel (same as --distributed), 2 : DataParallel (same as --parallel)  
  -pm, --pin_memory: Use pin_memory = True in the dataloader and set the output labels to cuda(NonBlocking=True)  
  -bb, --benchmark_backend: If False given, use torch.backends.cudnn.benchmark = False. Default: True  
  -ni, --num_images: Number of images in the random image dataset.  
  -ad, --average_duration: Calculate the average of the durations measured by each gpu. The duration is needed to get the images per second.  
  -ag, --average_gradients: Average the gradients of the model after each step on the cost of performance.  
  -pb, --process_group_backend: Choose a different backend for the distribution process group. "nccl" is supposed to have more features for distributed GPU training, but the performance on resnet50 seems to be better with "gloo". Default: "gloo".  