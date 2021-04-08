#!/usr/bin/env python3

import torch
import datetime
import time
import os
import sys
import torch.distributed as dist
import data.data as data
import models.model_utils as model_utils
import utils.utils as utils
import utils.flags as flags


def run(rank, img_data, data_name, gpu_info, args, start_time):
    """Run training/evaluation on given gpu id (rank).
    """
    total_step = len(img_data)
    model = model_utils.MultiGpuModel(rank, args.model, args.num_gpus, args.precision, args.parallel, args.eval)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=args.learning_rate, momentum=0.9)
    start = time.time()
    img_per_sec_dict = {}
    gpu_info_dict = {}
    gpu_temp_list = []

    if args.start_epoch == 0:
        args.start_epoch = model.check_saved_checkpoint_epoch(args.model, data_name) + 1
    #elif args.start_epoch != 1:
    #    args.start_epoch += 1

    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):

        if rank == 0:
            print(f'Epoch {epoch}')
        if epoch > 1:
            map_location = {'cuda:0': f'cuda:{rank}'}
            if not args.parallel:
                dist.barrier()
            if not args.eval:
                model.model, optimizer = model.load_model(optimizer, epoch-1, map_location)
        correct_predictions = 0
        total_predictions = 0
        for step, (data, label) in enumerate(img_data):
            try:
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
                if not args.eval:
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, label)
                    loss.backward()
                    #average_gradients(model, args.num_gpus)
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(data)
                        _, pred = torch.max(outputs.data, 1)
                        total_predictions += label.size(0)
                        correct_predictions += (pred == label).sum().item()
                        loss = 0.0
                start = utils.make_benchmark_calculations(
                    start, epoch, step, total_step, correct_predictions, total_predictions, loss, args, gpu_info,
                    rank, img_per_sec_dict, gpu_temp_list, gpu_info_dict
                                                          )
            except KeyboardInterrupt:
                if rank == 0:
                    utils.cancel_procedure(epoch, step, args, img_per_sec_dict, gpu_info_dict, gpu_temp_list, start_time)
        if rank == 0:
            if not args.eval:
                model.save_model(optimizer, epoch, data_name)

    print(*utils.make_protocol(img_per_sec_dict, gpu_info_dict, gpu_temp_list, args, start_time))
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
            print('You need to install the package nvidia-ml-py3 (f.i with "pip3 install nvidia-ml-py3") '
                  'to show gpu temperature and fan speed.\n')
    else:
        gpu_info = None

    torch.cuda.set_device(rank)
    if args.eval:
        _, img_data, label_dict, data_name = data.load_data(rank, args)
    else:
        img_data, _, label_dict, data_name = data.load_data(rank, args)
    run(rank, img_data, data_name, gpu_info, args, start_time)


def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.backends.cudnn.benchmark = True

    args = flags.load_flags()
    if args.pred_pic_label:
        utils.predict_picture_label(args.pred_pic_label)
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
