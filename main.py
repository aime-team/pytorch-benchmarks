#!/usr/bin/env python3

import torch
import sys
import torch.distributed as dist
from data.data import load_data
from models.utils import init_multi_gpu_model
import utils.utils as utils
from utils.flags import Flags


def run_training_process_on_given_gpu(rank, data):
    """Run training/evaluation on given gpu id (rank).
    """
    args = data.args
    model = init_multi_gpu_model(rank, args)
    protocol = utils.Protocol(rank, args, model, data)
    train_dataloader = data.get_train_dataloader()
    eval_dataloader = data.get_eval_dataloader()
    
    try:
        for epoch in range(1 + args.load_from_epoch, 1 + args.load_from_epoch + args.num_epochs):
            protocol.start_epoch(epoch)                  
            if not args.eval_only:
                protocol.set_to_train_mode()
                for step, batch in enumerate(train_dataloader, 1):
                    model_input, model_target_output = model.do_batch_processing(batch)
                    model.optimizer.zero_grad()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.auto_mixed_precision):
                        model_output = model(*model_input)
                        loss = model.do_backpropagation(model_output, model_target_output)
                        
                    if step % args.calc_every == 0 or step == data.total_steps_train:
                        protocol.show_progress(rank, epoch, step, data.total_steps_train, loss)
                protocol.finish_epoch(rank, epoch, model)

            if args.eval:
                protocol.set_to_eval_mode()

                with torch.no_grad():
                    for step, batch in enumerate(eval_dataloader, 1):
                        model_input, model_target_output = model.do_batch_processing(batch)
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.auto_mixed_precision):
                            model_output = model(*model_input)
                        protocol.evaluation.evaluate_step(model_output, model_target_output)
                        if step % args.calc_every == 0 or step == data.total_steps_eval:
                            protocol.show_progress(rank, epoch, step, data.total_steps_eval)                        
                protocol.finish_epoch(rank, epoch)                                         
        protocol.finish_benchmark()

    except KeyboardInterrupt:
        protocol.cancel_procedure()
    except RuntimeError as error_report:
        protocol.error_procedure(error_report)
        

def main():
    flags = Flags()
    args = flags.args
    try:
        if args.live_plot:
            _ = utils.run_live_plot_thread(args.num_gpus, args.refresh_interval)

        data = load_data(args)

        if args.pred_pic_label:
            model = init_multi_gpu_model(0, args)
            model.predict_label_for_single_picture()

        elif args.distributed:
            torch.multiprocessing.spawn(
            run_training_process_on_given_gpu, args=(data, ), nprocs=args.num_gpus, join=True
                                        )
        else:
            run_training_process_on_given_gpu(0, data)
    except KeyboardInterrupt:
        pass




if __name__ == '__main__':
    main()

