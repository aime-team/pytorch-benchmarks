#!/usr/bin/env python3

import torch
import datetime
import time
import os
import sys
import torch.distributed as dist
import data
import model_utils
import utils
import flags

torch.backends.cudnn.benchmark = True

def run(rank, img_data, gpu_info, args, start_time):
    """Run training/evaluation on given gpu id (rank).
    """
<<<<<<< HEAD
    total_step = len(img_data)
    multi_gpu_model = model_utils.MultiGpuModel(rank, args.model, args.num_gpus, args.precision, args.parallel)
    model = multi_gpu_model.model
    if args.eval:
        model.eval()
=======
    with open(f'{args.num_gpus}_{str(torch.cuda.get_device_name(0))}_{args.batch_size}.txt', 'w') as protocol:
        protocol.write(_info_text)
        for key in _img_per_sec_dict:
            protocol.write(f'Epoch: {key[0]}, Step: {key [1]}, Images per second: {_img_per_sec_dict[key]}\n')

        _mean_img_per_sec = np.array(list(_img_per_sec_dict.values())).mean()
        protocol.write(f'\nMean images per second: {_mean_img_per_sec}\n')

        _now = datetime.datetime.now()
        _end_time = _now.strftime('%Y/%m/%d %H:%M:%S')
        protocol.write(f'Benchmark end : {_end_time}\n')
        return _mean_img_per_sec


def load_data():
    """Loads the training data."""
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # args.img_folder = '~/gpu_benchmark/fruits-360/Training'

    if args.img_folder:
        # Real data
        img_dataset = torchvision.datasets.ImageFolder(root=args.img_folder, transform=transforms)
    else:
        # Random images
        image_size = (3, 224, 224)
        num_images = 100000
        num_classes = 1000
        img_dataset = RandomDataset(image_size, num_images, num_classes, transform=transforms)

    if args.num_workers == -1:
        num_workers = args.num_gpus
>>>>>>> 3d9bebadedd481fcff6a2ed65c0013f6d16c5269
    else:
        model.train()
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    start = time.time()
    img_per_sec_dict = {}
    gpu_info_dict = {}
    gpu_temp_list = []

    if args.start_epoch == 0:
        args.start_epoch = multi_gpu_model.check_pretrained_model_epoch(args.model)
    elif args.start_epoch != 1:
        args.start_epoch += 1

    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):

        if rank == 0:
            print(f'Epoch {epoch}')
        if epoch > 1:
            map_location = {'cuda:0': f'cuda:{rank}'}
            if not args.parallel:
                dist.barrier()
            model, optimizer = multi_gpu_model.load_model(optimizer, epoch-1, map_location)
        correct_predictions = 0
        total_predictions = 0
        for step, (_data, _label) in enumerate(img_data):
            try:
                _data, _label = _data.cuda(non_blocking=True), _label.cuda(non_blocking=True)
                if not args.eval:
                    optimizer.zero_grad()
                    outputs = model(_data)
                    loss = criterion(outputs, _label)
                    loss.backward()
                    #average_gradients(model, args.num_gpus)
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(_data)
                        _, pred = torch.max(outputs.data, 1)
                        total_predictions += _label.size(0)
                        correct_predictions += (pred == _label).sum().item()
                        loss = 0.0
                start = utils.make_benchmark_calculations(
                    start, epoch, step, total_step, correct_predictions, total_predictions, loss, args, gpu_info,
                    rank, img_per_sec_dict, gpu_temp_list, gpu_info_dict
                                                          )
                #if args.temp_test:
                #    utils.make_temp_test
            except KeyboardInterrupt:
                utils.cancel_procedure(epoch, step, args, img_per_sec_dict, gpu_info_dict, gpu_temp_list, start_time)
        if rank == 0:
            multi_gpu_model.save_model(optimizer, epoch)

    print(utils.make_protocol(img_per_sec_dict, gpu_info_dict, gpu_temp_list, args, start_time))
    if not args.parallel:
        dist.destroy_process_group()


def start_training(rank, args):
    """Prepare and start training for given gpu id (rank).
    """
    now = datetime.datetime.now()
    start_time = now.strftime('%Y/%m/%d %H:%M:%S')
    print(utils.make_info_text(args, start_time))
    if not args.parallel:
        dist.init_process_group(backend='gloo', rank=rank, world_size=args.num_gpus, init_method="env://")
    if rank == 0 and not args.no_temp:
        try:
            gpu_info = utils.GpuInfo(args.num_gpus)
        except AttributeError:
            gpu_info = None
            print('You need to install the package nvidia-ml-py3 (f.i with "pip3 install nvidia-ml-py3 '
                  'to show gpu temperature and fan speed.\n')
    else:
        gpu_info = None

    torch.cuda.set_device(rank)
    if args.eval:
        data_loader = data.load_data(rank, args)[1]
    else:
        data_loader = data.load_data(rank, args)[0]
    run(rank, data_loader, gpu_info, args, start_time)


def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.backends.cudnn.benchmark = True

    args = flags.load_flags()
    if args.live_plot:
        _ = utils.run_live_plot_thread(args.num_gpus, args.refresh_interval)

    try:
        if args.parallel:
            start_training(0, args)
        else:
            torch.multiprocessing.spawn(start_training, args=(args, ), nprocs=args.num_gpus, join=True)
    except KeyboardInterrupt:
        pass

    now = datetime.datetime.now()
    end_time = now.strftime('%Y/%m/%d %H:%M:%S')
    print(f'\nBenchmark end : {end_time}')


if __name__ == '__main__':
    main()
