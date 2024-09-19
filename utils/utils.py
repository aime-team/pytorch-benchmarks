#!/usr/bin/env python3

import torch
import platform
import sys
from datetime import datetime
import time
import torch.distributed as dist
import numpy as np
from pathlib import Path
from models.utils import MultiGpuModel
import subprocess
import threading
import csv
from collections import Counter, namedtuple
import json
import re
import string

EVAL_MODE_DICT = {True: 'evaluation', False: 'training'}

try:
    import nvidia_smi

    class GpuInfo(object):
        """Collects GPU information using nvidia-ml-py3.
        """

        def __init__(self, num_gpus):
            nvidia_smi.nvmlInit()
            self.num_gpus = num_gpus
            self.gpu_temp_list = []
            self.fan_speed_list = []
            self.gpu_usage_list = []
            self.memory_usage_list = []

        @staticmethod
        def get_driver_version():
            """Returns the installed nvidia driver version.
            """
            version = nvidia_smi.nvmlSystemGetDriverVersion()
            try:
                version = version.decode()
            except AttributeError:
                pass
            return version

        def get_current_attributes_all_gpus(self):
            """Returns tuple with list gpu attributes for all gpus.
            Updates all instance variables with the current state.
            """
            current_gpu_temp_list = []
            current_fan_speed_list = []
            current_gpu_usage_list = []
            current_memory_usage_list = []
            for gpu_id in range(self.num_gpus):
                gpu_temp, fan_speed, gpu_usage, memory_usage = GpuInfo.get_current_attributes(gpu_id)
                current_gpu_temp_list.append(gpu_temp)
                current_fan_speed_list.append(fan_speed)
                current_gpu_usage_list.append(gpu_usage)
                current_memory_usage_list.append(memory_usage)
            self.gpu_temp_list.append(current_gpu_temp_list)
            self.fan_speed_list.append(current_fan_speed_list)
            self.gpu_usage_list.append(current_gpu_usage_list)
            self.memory_usage_list.append(current_memory_usage_list)

            return current_gpu_temp_list, current_fan_speed_list, current_gpu_usage_list, current_memory_usage_list

        @staticmethod
        def get_current_attributes(gpu_id):
            """Returns tuple with gpu attributes for given gpu id.
            """
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)

            gpu_temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
            try:
                fan_speed = nvidia_smi.nvmlDeviceGetFanSpeed(handle)
            except nvidia_smi.NVMLError:
                fan_speed = 0

            gpu_usage = nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
            memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            memory_usage = memory_info.used, memory_info.total

            return gpu_temp, fan_speed, gpu_usage, memory_usage

        def get_gpu_info_str(self):
            """Returns gpu info string for all gpus.
            """
            current_gpu_temp_list, current_fan_speed_list, current_gpu_usage_list, current_memory_usage_list = self.get_current_attributes_all_gpus()
            gpu_info_str = ''
            for gpu_id in range(self.num_gpus):
                gpu_info_str += f'GPU-ID: {gpu_id}, Temperature: {current_gpu_temp_list[gpu_id]} °C, ' \
                                f'Fan speed: {current_fan_speed_list[gpu_id]}%, ' \
                                f'GPU usage: {current_gpu_usage_list[gpu_id]}%, ' \
                                f'Memory used: [{round((current_memory_usage_list[gpu_id][0] / 1024) / 1024 / 1024, 1)}' \
                                f'/ {round((current_memory_usage_list[gpu_id][1] / 1024) / 1024 / 1024, 1)}] GB\n'
            return gpu_info_str

        def get_list_of_max_temperatures_of_each_gpu(self):
            """Returns a list containing the maximum temperatures of all gpus ordered by rank.
            """
            if self.gpu_temp_list:
                gpu_temp_array = np.array(self.gpu_temp_list).transpose()
                return [max(gpu_temp_array[gpu_id]) for gpu_id in range(self.num_gpus)]

        def get_max_temperature_str(self):
            """Calculates the maximum temperature for each GPU and returns a string containing these infos.
            """
            max_temp_list = self.get_list_of_max_temperatures_of_each_gpu()
            
            if max_temp_list:
                max_temp_str = f'\nMaximum temperature(s): '
                for gpu_id, max_temp in enumerate(max_temp_list):
                    if not gpu_id == 0:
                        max_temp_str += '   ||  '
                    max_temp_str += f'GPU {gpu_id}: {max_temp} °C'
                return max_temp_str
            else:
                return ''

except ModuleNotFoundError:
    pass


class Benchmark(object):
    def __init__(self, rank, args):
        """Class for the benchmark.
        """
        self.args = args
        self.args.device
        self.start_time = time.time()
        self.epoch_start_time = self.start_time
        self.rank = rank
        self.gpu_temp_list = []
        self.it_per_sec_dicts = [{},{}]
        self.mean_it_per_sec_dicts_per_epoch = [{},{}]
        self.eval_mode = False
        self.warm_up_stage = True


    def check_if_warm_up_stage(self, epoch, step):
        """Check if the process is still in warm up stage and sets the boolean 
        variable self.warm_up_stage respectively. Returns self.warm_up_stage.
        """
        if step <= self.args.warm_up_steps:
            self.warm_up_stage = True
        else:
            self.warm_up_stage = False
        return self.warm_up_stage


    def calculate_iterations_per_sec(self, epoch, step):
        """Returns iterations per second for given arguments.
        """
        duration = self.get_duration()
        it_per_sec = self.args.batch_size * self.args.calc_every * self.args.num_gpus / duration
        if self.args.parallel:
            it_per_sec /= self.args.num_gpus
        if self.args.mean_it_per_sec and not self.warm_up_stage:
            self.it_per_sec_dicts[int(self.eval_mode)][step] = it_per_sec
        return it_per_sec


    def calculate_benchmark(self, epoch, step):
        """Calculates the benchmark values and prints status. 
        """
        prompt_str = ''
        self.check_if_warm_up_stage(epoch, step)
        it_per_sec = self.calculate_iterations_per_sec(epoch, step)
        
        return it_per_sec

    def get_duration(self):
        end_time = time.time()
        if self.args.average_duration:
            duration = torch.tensor(end_time - self.start_time).to(self.args.device)
            duration = self.average_duration(duration).detach().item()
        else:
            duration = end_time - self.start_time
        self.start_time = end_time
        return duration
    

    def make_epoch_mean_it_per_sec_string(self, epoch):
        """Calculates the mean iterations per second for training and/or evaluation for the last epoch and returns it
        in the shape of a printable string.
        """
        mean_it_per_sec = self.calc_mean_it_per_sec(self.it_per_sec_dicts[int(self.eval_mode)])
        self.it_per_sec_dicts[int(self.eval_mode)].clear()
        self.mean_it_per_sec_dicts_per_epoch[int(self.eval_mode)][epoch] = mean_it_per_sec
        return f'\nMean {self.args.iterations} per second in this {EVAL_MODE_DICT[int(self.eval_mode)]} epoch: {mean_it_per_sec}\n\n'


    def make_final_mean_it_per_sec_string(self):
        if self.mean_it_per_sec_dicts_per_epoch[0]:
            final_mean_it_per_sec_string_train = f'\nTotal mean {self.args.iterations.lower()} per second in training: {self.calc_mean_it_per_sec(self.mean_it_per_sec_dicts_per_epoch[0])}'
        elif self.it_per_sec_dicts[0]:
            final_mean_it_per_sec_string_train = f'\nTotal mean {self.args.iterations.lower()} per second in training: {self.calc_mean_it_per_sec(self.it_per_sec_dicts[0])}'
        else:
            final_mean_it_per_sec_string_train = ''
            
        if self.mean_it_per_sec_dicts_per_epoch[1]:
            final_mean_it_per_sec_string_eval = f'\nTotal mean {self.args.iterations.lower()} per second in evaluation: {self.calc_mean_it_per_sec(self.mean_it_per_sec_dicts_per_epoch[1])}'
        elif self.it_per_sec_dicts[1]:
            final_mean_it_per_sec_string_eval = f'\nTotal mean {self.args.iterations.lower()} per second in evaluation: {self.calc_mean_it_per_sec(self.it_per_sec_dicts[1])}'
        else:
            final_mean_it_per_sec_string_eval = '' 

        return final_mean_it_per_sec_string_train + final_mean_it_per_sec_string_eval + '\n'

    def calc_mean_it_per_sec(self, it_per_sec_dict):
        if len(it_per_sec_dict):
            mean_it_per_sec = round(np.array(list(it_per_sec_dict.values())).mean())
        else:
            mean_it_per_sec = 0
        return mean_it_per_sec


    def average_duration(self, duration):
        """Calculates the average of the duration over all gpus.
        """
        dist.barrier()
        dist.reduce(duration, 0, op=dist.ReduceOp.SUM)
        duration /= self.args.num_gpus
        return duration


class Protocol(object):

    def __init__(self, rank, args, model, data):
        """Class for the protocol.
        """
        self.rank = rank
        self.args = args 
        self.model = model   
        self.data = data
        self.epoch = 1 
        self.start_time = time.time()
        self.epoch_start_time = self.start_time
        self.total_evaluation_results = {}
        self.benchmark = self.init_benchmark()
        self.gpu_info = self.init_gpu_info()
        self.info_text = self.make_info_text()
        self.log_file = self.init_logfile()
        self.eval_mode = False
        self.evaluation = None
        self.error_report = ''


    def init_logfile(self):
        """Instantiates a Protocol object for GPU 0, if the flag --log_file is set.
        Returns the Protocol instance.
        """
        if self.rank == 0:
            if self.args.log_file:
                log_file = Logfile(self.args, self.info_text)
            else:
                log_file = None
        else:
            log_file = None
        return log_file

    def init_benchmark(self):
        if not self.args.no_benchmark:
            return Benchmark(self.rank, self.args)


    def init_gpu_info(self):
        """Instantiates a GpuInfo object, to measure GPU temps etc. if the flag --no_temp is not set.
        Returns the GpuInfo instance.
        """
        if self.rank == 0:
            if not self.args.no_temp:
                try:
                    gpu_info = GpuInfo(self.args.num_gpus)
                except AttributeError:
                    gpu_info = None
                    print('You need to install the package nvidia-ml-py3 (f.i with "pip3 install nvidia-ml-py3") '
                        'to show gpu temperature and fan speed.\n')
                except NameError:
                    gpu_info = None
                    print('You need to install the package nvidia-ml-py3 (f.i with "pip3 install nvidia-ml-py3") '
                        'to show gpu temperature and fan speed.\n')
            else:
                gpu_info = None
        else:
            gpu_info = None
        return gpu_info

    def start_epoch(self, epoch):
        if self.rank == 0:
            epoch_no_str = f'Epoch {epoch}\n'
            print(epoch_no_str)
            if self.args.log_file and self.args.log_benchmark:
                self.log_file.prompt_str_epoch += epoch_no_str
        self.epoch = epoch

    def set_to_train_mode(self):
        """Set the benchmark and the model to training mode.
        """
        self.eval_mode = False
        self.benchmark.eval_mode = False
        self.model.model.train()
        if self.args.distributed:
            self.data.train_sampler.set_epoch(self.epoch)


    def set_to_eval_mode(self):
        """Set the benchmark and the model to evaluation mode.
        """
        self.eval_mode = True
        self.benchmark.eval_mode = True
        self.model.model.eval()
        self.evaluation = init_evaluation(self.rank, self.args, self.data)


    def finish_epoch(self, rank, epoch, model=None):
        """Prints infos and writes all epoch plots into the protocol file after each epoch.
        """
        prompt_str = ''
        if self.eval_mode:
            evaluation_results_per_epoch = self.evaluation.evaluate_epoch()

        if self.rank == 0:
            if self.gpu_info:
                prompt_str += self.gpu_info.get_max_temperature_str()
            if not self.eval_mode:
                prompt_str += '\nTraining '
                if not self.args.skip_checkpoint:
                    model.save_checkpoint(epoch, self.args.data_name)
            else:
                for key, value in evaluation_results_per_epoch.items():
                    prompt_str += f'\n{key}:{value:.4f}'
                    
                prompt_str+='\n\nEvaluation '
                self.total_evaluation_results[epoch] = evaluation_results_per_epoch
                
                
            prompt_str += f'epoch finished within {self.make_epoch_duration_string()}.\n'
            if self.args.mean_it_per_sec:
                prompt_str += self.benchmark.make_epoch_mean_it_per_sec_string(epoch)
            print(prompt_str, end='')
            
            if self.args.log_file:
                self.log_file.prompt_str_epoch += prompt_str
                if self.args.log_benchmark:
                    self.log_file.update_log_file()
                self.log_file.prompt_str_epoch = ''
                if self.eval_mode:
                    self.log_file.update_csv_file(epoch, evaluation_results_per_epoch.values())
        return True


    def finish_benchmark(self):
        """Writes benchmark results in a text file and calculates the mean.
        Returns the mean iterations per sec.
        """
        if self.rank == 0:
            if self.args.mean_it_per_sec:
                mean_it_per_sec_str = self.benchmark.make_final_mean_it_per_sec_string()
            else:
                mean_it_per_sec_str = ''
            if self.gpu_info:
                max_temp_str = self.gpu_info.get_max_temperature_str()
            else:
                max_temp_str = ''
            total_evaluation_results_str = ''
            if self.total_evaluation_results:
                for epoch, evaluation_results_per_epoch in self.total_evaluation_results.items():
                    total_evaluation_results_str += f'\nEpoch {epoch}: '
                    for key, value in evaluation_results_per_epoch.items():
                        total_evaluation_results_str += f'{key}: {value}   '

            now = datetime.now()
            end_time = now.strftime('%Y/%m/%d %H:%M:%S')
            final_str = mean_it_per_sec_str + max_temp_str + total_evaluation_results_str + self.error_report + f'\n\nBenchmark end: {end_time}\n'
            if self.args.log_file:
                self.log_file.finish_log_file(final_str)
            print(final_str)
        #dist.barrier()
        if self.args.distributed:
            dist.destroy_process_group()

        return True

    def show_progress(self, rank, epoch, step, total_steps, loss=None):
        if rank == 0:
            if not self.args.no_benchmark and step != total_steps:
                it_per_sec = self.benchmark.calculate_benchmark(epoch, step)
            else:
                it_per_sec= None
            progress_prompt_str = self.make_progress_prompt_string(epoch, step, total_steps, loss, it_per_sec)

            if self.gpu_info:
                progress_prompt_str += self.gpu_info.get_gpu_info_str()

            print(progress_prompt_str, end = '')
            if self.args.log_file and self.args.log_benchmark:
                self.log_file.prompt_str_epoch += progress_prompt_str


    def make_progress_prompt_string(self, epoch, step, total_steps, loss=None, it_per_sec=None):
        """Creates and returns the benchmark prompt string for given arguments.
        """
        progress_prompt_string = f'Epoch [{epoch} / {self.args.load_from_epoch + self.args.num_epochs}], Step [{step} / {total_steps}]'
        if not self.eval_mode:
            loss_item = loss.detach().item()
            progress_prompt_string += f', Loss: {loss_item:.4f}'

        if it_per_sec:
            progress_prompt_string += f',  {self.args.iterations} per second: {it_per_sec:.1f}'
        progress_prompt_string += '\n'
        return progress_prompt_string


    def cancel_procedure(self):
        """Called in the case of KeyboardInterrupt.
        Finishes the protocol, prints mean and temperature maxima and exits the program.
        """
        if self.rank == 0:
            if self.args.log_file:
                self.log_file.update_log_file()
        self.finish_benchmark()
        if self.rank == 0:
            sys.exit('Cancelled by KeyboardInterrupt\n')

    def error_procedure(self, local_error_report, queue):
        if self.rank != 0:
            queue.put(local_error_report)
        if self.rank == 0:
            time.sleep(0.5)
            self.error_report = str(local_error_report)+'\n'
            if self.args.distributed:
                if not queue.empty():
                    self.error_report += str(queue.get())

        self.finish_benchmark()


    def make_info_text(self):
        """Makes info text about the device, the OS and some parameters, 
        shown at the beginning of the benchmark and in the protocol.
        """
        if self.rank == 0:
            start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            cpu_name = 'unknown'
            for line in subprocess.check_output("lscpu", shell=True).strip().decode().split('\n'):
                if 'Modellname' in line or 'Model name' in line:
                    cpu_name = line.split(':')[1].strip()
            dist_dict = {
                0 : 'Single GPU Training ', 
                1 : 'Distributed Data Parallel', 
                2 : 'Data Parallel'
                         }

            if self.args.dist_optim:
                optimizer = 'ZeroRedundancyOptimizer with SGD'
            elif self.args.dist_optim_190:
                optimizer = 'ZeroRedundancyOptimizer with SGD from Pytorch 1.9.0'
            else:
                optimizer = self.args.optimizer

            info_text = f'OS: {platform.uname().system}, {platform.uname().release}\n'\
                        f'Device-name: {platform.uname().node}\n'\

            if self.args.use_cpu:
                info_text += f'CPU used for benchmark: {cpu_name}\n'
            else:
                info_text += f'{self.args.num_gpus} GPU(s) used for benchmark:\n'
                for gpu_id in range(self.args.num_gpus):
                    gpu_name = str(torch.cuda.get_device_name(gpu_id))
                    info_text += f'{gpu_id}: {gpu_name}\n'
            if not self.args.no_temp:
                try:
                    info_text += f'Nvidia GPU driver version: {GpuInfo.get_driver_version()}\n'
                except NameError:
                    pass
            info_text += f'Available GPUs on device: {torch.cuda.device_count()}\n' \
                         f'Cuda-version: {torch.version.cuda}\n' \
                         f'Cudnn-version: {torch.backends.cudnn.version()}\n' \
                         f'Python-version: {sys.version.split(" ")[0]}\n' \
                         f'PyTorch-version: {torch.__version__}\n' \
                         f'CPU: {cpu_name}\n' \
                         f'Model: {self.args.model}\n' \
                         f'Global train batch size: {self.args.global_batch_size}\n' \
                         f'Local train batch size: {self.args.batch_size}\n' \
                         f'Global evaluation batch size: {self.args.global_eval_batch_size}\n' \
                         f'Local evaluation batch size: {self.args.eval_batch_size}\n' \
                         f'Distribution Mode: {dist_dict[self.args.distribution_mode]}\n' \
                         f'Process group backend: {self.args.process_group_backend}\n' \
                         f'Optimizer: {optimizer}\n' \
                         f'Precision: {"Automatic mixed precision" if self.args.auto_mixed_precision else self.args.precision}\n' \
                         f'Compile-Mode: {self.args.compile}\n' \
                         f'Log file: {self.args.log_file}\n' \
                         f'Training data: {self.args.data_name}\n' \
                         f'Initial learning rate: {self.args.learning_rate}\n' \
                         f'Learning rate decay step: {self.args.step_lr}\n' \
                         f'Used data augmentation: {not self.args.no_augmentation}\n' \
                         f'Checkpoint folder: {self.args.checkpoint_folder}\n' \
                         f'Number of workers: {self.args.num_workers}\n' \
                         f'Warm up steps: {self.args.warm_up_steps}\n' \
                         f'Benchmark start : {start_time}\n\n'
            print(info_text, end='')
            return info_text
        return None

    def make_epoch_duration_string(self):
        """Makes a duration string splitted in hours/minutes/seconds for
        given duration in seconds.
        """
        epoch_duration = self.calculate_epoch_duration()
        epoch_duration_str = ''
        hours = epoch_duration // 3600
        minutes = epoch_duration % 3600 // 60
        seconds = epoch_duration % 60
        if hours:
            epoch_duration_str += f'{hours} hours, '
        if minutes:
            epoch_duration_str += f'{minutes:.0f} minutes and '
        epoch_duration_str += f'{seconds:.0f} seconds'
        return epoch_duration_str

    def calculate_epoch_duration(self):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_start_time = epoch_end_time
        return epoch_duration


class Logfile(object):
    """Initializes the Protocol.
    """
    def __init__(self, args, info_text):
        self.args = args
        self.log_file = self.init_log_file(info_text)
        self.init_csv_file()
        self.prompt_str_epoch = ''



    def init_log_file(self, info_text):
        """Initializes the log file and writes the info text into it.
        Returns the PosixPath of the log file's destination.
        """
        log_dir = Path(__file__).absolute().parent.parent / 'log'
        if not log_dir.is_dir():
            log_dir.mkdir()
        log_file = log_dir / self.args.log_file

        if self.args.load_from_epoch == 0:
            with open(log_file, 'w') as protocol:
                protocol.write(info_text)
        else:
            with open(log_file, 'a') as protocol:
                protocol.write(info_text)
        return log_file

    def update_log_file(self):
        """Writes the all the prompts of the current epoch into the protocol file.
        """
        with open(self.log_file, 'a') as log_file:
            log_file.write(self.prompt_str_epoch)
        return True

    def init_csv_file(self):
        if self.args.load_from_epoch == 0:
            with open(self.log_file.with_suffix('.csv'), 'w') as f:
                writer = csv.writer(f)
        else:
            with open(self.log_file.with_suffix('.csv'), 'a') as f:
                writer = csv.writer(f)

    def update_csv_file(self, epoch, evaluation_results_per_epoch):
        with open(self.log_file.with_suffix('.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((epoch, *(result for result in evaluation_results_per_epoch)))

    def finish_log_file(self, final_str):
        """Writes given string with benchmark end results like mean iterations per second, maximum GPU temperatures 
        and the validation accuracy from each epoch into the protocol file.
        """
        with open(self.log_file, 'a') as log_file:
            log_file.write(final_str)
        return True


class EvaluationImageModel(object):
    def __init__(self, rank, args):
        self.rank = rank
        self.args = args
        torch.cuda.set_device(rank)
        self.correct_predictions = torch.tensor(0).to(args.device)
        self.correct_predictions5 = torch.tensor(0).to(args.device)
        self.total_predictions = torch.tensor(0).to(args.device)


    def evaluate_step(self, predicted_label, label):
        _, pred = torch.max(predicted_label.data, 1)
        _, pred5 = predicted_label.topk(5, 1, True, True)
        label_resize = label.view(-1,1)
        self.total_predictions += label.size(0)
        self.correct_predictions += (pred == label).sum()
        self.correct_predictions5 += (pred5 == label_resize).sum()

    def evaluate_epoch(self):
        self.sum_up_predictions_of_all_gpus()
        val_acc, val_acc5 = self.calculate_validation_accuracy()
        return {'Validation accuracy': val_acc, 'Validation accuracy top5': val_acc5}

    def sum_up_predictions_of_all_gpus(self):
        """Sums up the correct predictions and the total predictions of the evaluation on each gpu.
        """
        if self.args.distributed:
            dist.barrier()
            dist.all_reduce(self.correct_predictions, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.correct_predictions5, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total_predictions, op=dist.ReduceOp.SUM)

    def calculate_validation_accuracy(self):
        """Return the validation accuracy.
        """
        val_acc = self.correct_predictions / self.total_predictions
        val_acc5 = self.correct_predictions5 / self.total_predictions

        return round(val_acc.item(), 4), round(val_acc5.item(), 4)


class EvaluationBert(object):
    """Evaluation of BERT
    """
    def __init__(self, rank, args, data):
        self.rank = rank
        self.all_results = []
        self.args = args
        self.data = data
        self.num_gpus = self.args.num_gpus


    def evaluate_step(self, model_output, example_indices):
        for i, example_index in enumerate(example_indices[0]):
            if not self.args.synthetic_data:
                eval_features = self.data.preprocessed_data.eval_features[example_index.item()]
                unique_id = torch.tensor(eval_features.unique_id).to(self.args.device)
            else:
                unique_id = torch.randint(low=0, high=self.args.num_synth_data, size=[1], dtype=torch.long).to(self.args.device)
            
            start_logits = model_output[0][i].contiguous()
            end_logits = model_output[1][i].contiguous()
            all_gpu_unique_id = [torch.zeros_like(unique_id) for _ in
                            range(self.num_gpus)]
            all_gpu_start_logits = [torch.zeros_like(start_logits) for _ in
                            range(self.num_gpus)]
            all_gpu_end_logits = [torch.zeros_like(end_logits) for _ in
                            range(self.num_gpus)]
            
            if self.args.distributed:
                dist.all_gather(all_gpu_unique_id, unique_id)
                dist.all_gather(all_gpu_start_logits, start_logits)
                dist.all_gather(all_gpu_end_logits, end_logits)
                if self.rank == 0:
                    raw_result = namedtuple("RawResult",
                        ["unique_id", "start_logits", "end_logits"])
                    for i in range(self.num_gpus):
                        unique_id = int(all_gpu_unique_id[i].detach())#.cpu())
                        start_logits = all_gpu_start_logits[i].detach().tolist()#.cpu().tolist()
                        end_logits = all_gpu_end_logits[i].detach().tolist()#cpu().tolist()
                        self.all_results.append(raw_result(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
            else:
                raw_result = namedtuple("RawResult",
                        ["unique_id", "start_logits", "end_logits"])
                self.all_results.append(raw_result(unique_id=unique_id.detach(), start_logits=start_logits.detach().tolist(), end_logits=end_logits.detach().tolist()))

    def evaluate_epoch(self):
        
        if self.rank == 0:
            answers, nbest_answers = self.data.preprocessed_data.get_answers(self.all_results)
            with open(self.args.eval_data_file) as eval_data_file:
                dataset_json = json.load(eval_data_file)
            dataset = dataset_json['data']
            self.all_results = []
            return EvaluationBert.get_f1_and_exact_match(dataset, answers)


    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = EvaluationBert.normalize_answer(prediction).split()
        ground_truth_tokens = EvaluationBert.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return (EvaluationBert.normalize_answer(prediction) == EvaluationBert.normalize_answer(ground_truth))

    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    @staticmethod
    def get_f1_and_exact_match(dataset, predictions):
        f1 = exact_match = total = 0
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                ' will receive score 0.'
                        #print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    exact_match += EvaluationBert.metric_max_over_ground_truths(
                        EvaluationBert.exact_match_score, prediction, ground_truths)
                    f1 += EvaluationBert.metric_max_over_ground_truths(
                        EvaluationBert.f1_score, prediction, ground_truths)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {'exact_match': round(exact_match, 4), 'f1': round(f1, 4)}




def run_live_plot_thread(num_gpus, refresh_interval=500):
    """ Runs the python-script gpu_live_plot.py via subprocess.
    """
    cwd = Path.cwd()
    gpu_live_plot_file = cwd / 'utils' / 'gpu_live_plot.py'
    thread = threading.Thread(
        target=subprocess.run, args=(
            ['python3', gpu_live_plot_file, '--num_gpus', str(num_gpus), '--refresh_interval', str(refresh_interval)],
                                     )
                              )
    thread.daemon = True
    thread.start()
    return thread



def init_evaluation(rank, args, data):
    if args.bert:
        evaluation = EvaluationBert(rank, args, data)
    else:
        evaluation = EvaluationImageModel(rank, args)
    return evaluation


