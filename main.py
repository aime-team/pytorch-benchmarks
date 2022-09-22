#!/usr/bin/env python3

import torch
import sys
import torch.distributed as dist
from data.data import MultiGpuData
from models.utils import MultiGpuModel
import utils.utils as utils
import utils.flags as flags


def run_training_process_on_given_gpu(rank, bench_data):
    """Run training/evaluation on given gpu id (rank).
    """
    args = bench_data.args
    model = MultiGpuModel(rank, args)
    benchmark = utils.Benchmark(rank, args)
    train_dataloader, sampler = bench_data.get_train_dataloader(rank)
    val_dataloader, _ = bench_data.get_val_dataloader(rank)
    total_steps_train = len(train_dataloader)
    total_steps_val = len(val_dataloader)
    
    try:
        for epoch in range(1 + args.load_from_epoch, 1 + args.load_from_epoch + args.num_epochs):
            if rank == 0:
                epoch_str = f'\nEpoch {epoch}\n'
                print(epoch_str)
                if args.log_file:
                    benchmark.protocol.prompt_str_epoch += epoch_str

            if epoch > 1:
                model.load_pretrained_model(epoch-1, bench_data.data_name, rank) 
                        
            if not args.eval_only:
                benchmark.set_to_train_mode(model)
                if args.distributed:
                    sampler.set_epoch(epoch)
                for step, (data, label) in enumerate(train_dataloader, 1):
                    data, label = data.cuda(rank, non_blocking=args.pin_memory), label.cuda(rank, non_blocking=args.pin_memory)  
                    if args.model == 'perceiver':
                        data = data.transpose(1,3).transpose(1,2)

                    model.optimizer.zero_grad()
                    outputs = model(data)
                    loss = model.criterion(outputs, label)
                    loss.backward()
                    if args.average_gradients and args.distributed:
                        model.average_gradients()
                    model.optimizer.step()
                    if step % args.calc_every == 0:
                        benchmark.calculate_benchmark(rank, epoch, step, total_steps_train, loss)

                benchmark.finish_epoch(rank, epoch, total_steps_train, loss)
                model.scheduler.step()
                if rank == 0:
                    model.save_model(epoch, bench_data.data_name)

            if args.eval or args.eval_only:
                benchmark.set_to_eval_mode(model)
                with torch.no_grad():
                    for step, (data, label) in enumerate(val_dataloader, 1):
                        data, label = data.cuda(rank, non_blocking=args.pin_memory), label.cuda(rank, non_blocking=args.pin_memory)
                        if args.model == 'perceiver':
                            data = data.transpose(1,3).transpose(1,2)
                        outputs = model(data)
                        loss = model.criterion(outputs, label)
                        _, pred = torch.max(outputs.data, 1)
                        label_resize = label.view(-1,1)
                        _, pred5 = outputs.topk(5, 1, True, True)
                        benchmark.total_predictions += label.size(0)
                        benchmark.correct_predictions += (pred == label).sum().cuda(rank)
                        benchmark.correct_predictions5 += (pred5 == label_resize).sum().cuda(rank)
                        if step % args.calc_every == 0:
                            benchmark.calculate_benchmark(rank, epoch, step, total_steps_val, loss)                        
                benchmark.finish_epoch(rank, epoch, total_steps_val, loss)
        if rank == 0:                                                        
            benchmark.finish_benchmark()
        if args.distributed:
            dist.destroy_process_group()
    except KeyboardInterrupt:
        if rank == 0:
            benchmark.cancel_procedure()


def main():
    args = flags.load_flags()
    try:
        if args.live_plot:
            _ = utils.run_live_plot_thread(args.num_gpus, args.refresh_interval)

        bench_data = MultiGpuData(args)
        if args.pred_pic_label:
            model = MultiGpuModel(0, args)
            model.predict_label_for_single_picture()

        elif args.distributed:
            torch.multiprocessing.spawn(
            run_training_process_on_given_gpu, args=(bench_data, ), nprocs=args.num_gpus, join=True
                                        )
        else:
            run_training_process_on_given_gpu(0, bench_data)
    except KeyboardInterrupt:
        pass




if __name__ == '__main__':
    main()
