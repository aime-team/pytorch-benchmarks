# PyTorch-Benchmarks

Tool to benchmark multi-gpu systems with PyTorch.  
  
## Usage
Run main.py to start benchmarking!

## Optional arguments:
  --warm_up_steps, -w : Number of warm up steps in every epoch. Warm up steps will not taken into account; default: 10  
  --num_epochs, -n : Number of epochs; default: 10  
  --batch_size, -b : Batch size; default: 64  
  --num_gpus, -g :Number of gpus used for training; default: 1  
  --gpu_ids , -i : IDs of used GPUs for training. If not given, range(num_gpus - 1) is used.  
  --model, -m : Model used for training; default: resnet50  
  --use_fp16, -f : Use half precision; default: False  
  --img_folder , -imf: Destination of training images. If not given, random data is used.  
  --num_workers, -nw : Number of workers used by the dataloader. If not given, num_gpus is used.
