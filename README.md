# PyTorch-Benchmarks

Benchmark tool for multiple models on multi-GPU setups. Compatible to CUDA (NVIDIA) and ROCm (AMD).


## Models  
Available and tested:
- bert-large-cased, bert-large-uncased, bert-base-cased, base-base-uncased
- resnet50, resnet101, resnet152  
- vgg16, vgg19  
- efficientnet_b0 - efficientnet_b5  
- densenet121, densenet169, densenet201  
  
Pretrained versions are implemented for the models bert-large-cased, bert-large-uncased, bert-base-cased, base-base-uncased and resnet50. Use them by adding the flag --pretrained.

## Usage  

A benchmark training on 1 GPU with the default training batch size of 64 using the default model `ResNet-50` with synthetic data is started with the following command:  
```
python3 main.py
```

A training of the model `ResNet-50` on 2 GPUs with a local batch size of 192 on your own training image dataset and evaluation after each epoch with a seperated test dataset is started with:  
```
python3 main.py --num_gpus 2 --batch_size 192 --eval --train_image_folder /destination/of/your/training/dataset --eval_image_folder /destination/of/your/evaluation/dataset
```

The training of the language model `BERT large uncased` on 4 GPUs with a global training batch size of 48 on the included SQuAD dataset is started with:  
```
python3 main.py --num_gpus 4 --model bert-large-uncased --data_name squad --global_batch_size 48 
```

If you want to evaluate it with the included SQuAD test dataset after each epoch with a global evaluation batch size of 64 
```
python3 main.py --num_gpus 4 --model bert-large-uncased --data_name squad --global_batch_size 48 --eval --global_eval_batch_size 64
```

If --skip_checkpoint is not set, the model state is saved after each training epoch under --checkpoint_folder. To continue a previous training from a certain epoch, use --load_from_epoch <epoch_no>. F.i.:  
```
python3 main.py --num_gpus 4 --model bert-large-uncased --data_name squad --global_batch_size 48 --eval --global_eval_batch_size 64 --load_from_epoch 10
```

To use the with PyTorch 2 newly implemented feature `model = torch.compile(model)` simply add --compile to your comand:
```
python3 main.py --compile  
```

The default numerical precision is fp32. To use automatic mixed precision (amp) to get a performance boost, add the flag -amp.

```
python3 main.py -amp  
```

By default the module torch.nn.parallel.DistributedDataParallel is used for Multi GPU training. To use the module torch.nn.parallel.DataParallel add the flag --parallel

```
python3 main.py --parallel  
```
        
  
## Optional arguments:  
  
  **-ne, --num_epochs** : Number of epochs. Default: 10  
  **-b, --batch_size** : Local batch size for training/evaluation. Default: 64  
  **-gb, --global_batch_size** : Global batch size for training. Default: 64 * num_gpus  
  **-eb, --eval_batch_size** : Local batch size for evaluation. Default: batch size for training: 64  
  **-geb, --global_eval_batch_size** : Global batch size for evaluation. Default: global batch size for training: 64 * num_gpus  
  **-ng, --num_gpus** : Number of gpus used for training. Default: 1  
  **-m, --model** : Model used for training. Default: resnet50  
  **-pt2, --compile** : Do optimizations for Pytorch 2. Does not work with Pytorch 1.  
  **-ev, --eval** : If given, the model is evaluated with the given validation dataset in --eval_folder or --eval_data_file at the end of each epoch and prints the evaluation accuracy.  
  **-eo, --eval_only** : If given, the model is evaluated with the given validation dataset without any training.  
  **-nw, --num_workers** : Number of workers used by the dataloader. If not given, 4 \* num_gpus is used.  
  **-amp, --auto_mixed_precision** : Enable automatic mixed precision.  
  **-dp, --parallel** : Use DataParallel for Multi-GPU training instead of DistributedDataParallel. If not given, DistributedDataParallel is used.  
  **-ddp, --distributed** : Use DistributedDataParallel even for single GPU training.  
  **-tf, --train_image_folder** : Destination of the training image dataset. If not given, synthetic data is used.  
  **-ef, --eval_image_folder** : Destination of the validation image dataset. If not given, synthetic data is used.  
  **-lr, --learning_rate** : Set the learning rate for training. Default: 1e-3  
  **-slr, --step_lr** : Decay the learning rate by factor 10 every given epoch. Default: 30  
  **-ldf, --lr_decay_factor** : Change the factor of the learning rate decay. Default: 10  
  **-cl, --constant_learning_rate** : Train with a constant learning rate  
  **-sd, --split_data** : Splits the given training dataset in a training and evaluation dataset with the given ratio. Takes values between 0 and 1. Default: 1.  
  **-le, --load_from_epoch**: Loads model state at given epoch of a saved checkpoint given in --checkpoint_folder. . If -1 is given, the highest available epoch for the model and the dataset is used.  
  **-w, --warm_up_steps** : Number of warm up steps in every epoch. Warm up steps will not taken into account. Default: 10  
  **-nt, --no_temp** : Hide GPU infos like temperature etc. for better performance.  
  **-f, --use_fp16** : Use half precision. If not given fp32 precision is used.  
  **-mi, --mean_it_per_sec** : Plot mean images/sequences per second at the end and save it in the logfile.  
  **-s, --set_seed** : Set the random seed. Default: 1234  
  **-ce, --calc_every** : Set the stepsize for calculations and print outputs. Default: 10  
  **-lp, --live_plot** : Show live plot of gpu temperature and fan speed.  
  **-ri, --refresh_interval** : Change live plot refresh interval in ms. Default: 500  
  **-pl, --pred_pic_label** : Predict label of given picture with a pretrained model.  
  **-in, --imagenet** : Use imagenet for training/evaluation from given path.  
  **-ln, --log_file** : Save the logfile in /log/ under given name. If no name is given, '<num_gpus>_<GPU_name>_<model_name>_<batch_size>_<learning_rate>' is used.  
  **-cf, --checkpoint_folder** : Save training checkpoints in given folder name.  If not given, the name of the log-file is used.  
  **-cfi, --checkpoint_file** : Load checkpoint from given file.  
  **-op, --optimizer** : Set the optimizer. Default: SGD  
  **-do, --dist_optim** : Use distributed optimizer (ZeroRedundancyOptimizer). (Experimental)  
  **-do9, --dist_optim_190** : Use distributed Optimizer (ZeroRedundancyOptimizer) from torch.distributed.optim (available in Pytorch 1.9.0., experimental).  
  **-dm, --distribution_mode** : Set distribution mode: 0 : None, 1: DistributedDataParallel (same as --distributed), 2 : DataParallel (same as --parallel)  
  **-pm, --pin_memory** : Use pin_memory = True in the dataloader and set the output labels to cuda(NonBlocking=True)  
  **-bb, --benchmark_backend** : Use torch.backends.cudnn.benchmark = True.  
  **-syn, --synthetic_data** : Use synthetic data. Default: False (True, if no data is given)  
  **-ni, --num_synth_data** : Number of examples in the synthetic dataset. Default: 100000  
  **-ad, --average_duration** : Calculate the average of the durations measured by each gpu. The duration is needed to get the images per second.  
  **-ag, --average_gradients** : Average the gradients of the model after each step on the cost of performance (Experimental, no improvement in training).  
  **-pb, --process_group_backend** : Choose a different backend for the distribution process group (nccl or gloo). "nccl" is supposed to have more features for distributed GPU training. Default: "nccl"  
  **-lb, --log_benchmark** : Write all the benchmark results into the log file.  
  **-na, --no_augmentation** : No augmentation of the image training dataset.  
  **-pt, --pretrained** : Load pretrained model from --checkpoint_file.  
  **-bt, --benchmark_train** : Start training for benchmark.  
  **-bts, --benchmark_train_steps** : Number of steps for --benchmark_train.  
  **-dn, --data_name** : Own name of dataset used for training/evaluation, if not ImageNet or Squad is used.  
  **-tdf, --train_data_file** : For BERT: SQuAD json for training. E.g., train_squad.json  
  **-edf, --eval_data_file** : For BERT: SQuAD json for evluation. E.g., eval_squad.json  
  **-msl, --max_seq_length** : For BERT: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Default=384  
  **-ds, --doc_stride** : For BERT: When splitting up a long document into chunks, how much stride to take between chunks. Default=128  
  **-mql, --max_query_length** : For BERT: The maximum number of tokens for the question. Questions longer than this will be truncated to this length. Default: 64  
  **-wp, --warmup_proportion** : For BERT: Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training. Default : 0.1  
  **-nbs, --n_best_size** : For BERT: The total number of n-best predictions. Default: 20  
  **-mal, --max_answer_length** : For BERT: The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another. Default: 30  
  **-gas, --gradient_accumulation_steps** : Number of updates steps to accumulate before performing a backward/update pass. Default: 1  
  **-dlc, --do_lower_case** : For BERT: Whether to lower case the input text. True for uncased models, False for cased models.  
  **-v2n, --version_2_with_negative** : If true, the SQuAD examples contain some that do not have an answer.  
  **-nsd, --null_score_diff_threshold** : If null_score - best_non_null is greater than the threshold predict null. Default: 0.0  
  **-vf, --vocab_file** : For BERT: Vocabulary mapping/file BERT was pretrainined on.  
  **-sc, --skip_checkpoint** : Whether to save checkpoints  
  **-sca, --skip_cache** : For BERT: Whether to cache train/evaluation features  
  **-ctf, --cached_train_features_file** : For BERT: Location to cache train feaures. Will default to the dataset directory.  
  **-cef, --cached_eval_features_file** : For BERT: Location to cache evaluation feaures. Will default to the dataset directory.  
  **-rnc, --renew_cache** : For BERT: Rebuild cache file of training/evaluation data.  
  **-se, --seed** : Random seed for initialization. Default : 1234  
  **-nb, --no_benchmark** : Don't do benchmark calculations.  
  **-up, --find_unused_parameters** : Enabling unused parameter detection by passing the keyword argument "find_unused_parameters=True" to `torch.nn.parallel.DistributedDataParallel.  
  **-st, --stress** : Stresstest with an infinite dataloader.  
  **-cpu, --use_cpu** : Use the cpu instead of gpus.  
