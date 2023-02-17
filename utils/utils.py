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

try:
    import nvidia_smi

    class GpuInfo(object):
        """Collects GPU information using nvidia-ml-py3.
        """

        def __init__(self, num_gpus):
            nvidia_smi.nvmlInit()
            self.num_gpus = num_gpus
            self.gpu_temp_list = [0 for gpu_id in range(num_gpus)]
            self.fan_speed_list = [0 for gpu_id in range(num_gpus)]
            self.gpu_usage_list = [0 for gpu_id in range(num_gpus)]
            self.memory_usage_list = [0 for gpu_id in range(num_gpus)]

        @staticmethod
        def get_driver_version():
            """Returns the installed nvidia driver version.
            """
            return nvidia_smi.nvmlSystemGetDriverVersion()

        def get_current_attributes_all_gpus(self):
            """Returns tuple with list gpu attributes for all gpus.
            Updates all instance variables with the current state.
            """
            gpu_temp_list = []
            fan_speed_list = []
            gpu_usage_list = []
            memory_usage_list = []
            for gpu_id in range(self.num_gpus):
                gpu_temp, fan_speed, gpu_usage, memory_usage = GpuInfo.get_current_attributes(gpu_id)
                gpu_temp_list.append(gpu_temp)
                fan_speed_list.append(fan_speed)
                gpu_usage_list.append(gpu_usage)
                memory_usage_list.append(memory_usage)
            self.gpu_temp_list = gpu_temp_list
            self.fan_speed_list = fan_speed_list
            self.gpu_usage_list = gpu_usage_list
            self.memory_usage_list = memory_usage_list

            return gpu_temp_list, fan_speed_list, gpu_usage_list, memory_usage_list

        @staticmethod
        def get_current_attributes(gpu_id):
            """Returns tuple with gpu attributes for given gpu id.
            """
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)

            gpu_temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
            fan_speed = nvidia_smi.nvmlDeviceGetFanSpeed(handle)
            gpu_usage = nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
            memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            memory_usage = memory_info.used, memory_info.total

            return gpu_temp, fan_speed, gpu_usage, memory_usage

        def get_gpu_info_str(self):
            """Returns gpu info string for all gpus.
            """
            gpu_info_str = ''
            for gpu_id in range(self.num_gpus):
                gpu_info_str += f'GPU-ID: {gpu_id}, Temperature: {self.gpu_temp_list[gpu_id]} °C, ' \
                                f'Fan speed: {self.fan_speed_list[gpu_id]}%, ' \
                                f'GPU usage: {self.gpu_usage_list[gpu_id]}%, ' \
                                f'Memory used: [{round((self.memory_usage_list[gpu_id][0] / 1024) / 1024 / 1024, 1)}' \
                                f'/ {round((self.memory_usage_list[gpu_id][1] / 1024) / 1024 / 1024, 1)}] GB\n'
            return gpu_info_str

except ModuleNotFoundError:
    pass


class Benchmark(object):
    def __init__(self, rank, args):
        """Class for the benchmark.
        """
        self.args = args
        self.start_time = time.time()
        self.epoch_start_time = self.start_time
        self.rank = rank
        self.gpu_temp_list = []
        self.img_per_sec_dicts = [{},{}]
        self.val_acc_dict = {}
        self.val_acc5_dict = {}
        self.gpu_info = self.init_gpu_info()
        self.info_text = self.make_info_text()
        self.protocol = self.init_protocol()
        self.warm_up_stage = True
        self.eval_mode = False
        self.correct_predictions = torch.tensor(0).cuda()
        self.correct_predictions5 = torch.tensor(0).cuda()
        self.total_predictions = torch.tensor(0).cuda()
        self.epoch_mean_img_per_sec_dicts = [{},{}]

    def init_protocol(self):
        """Instantiates a Protocol object for GPU 0, if the flag --log_file is set.
        Returns the Protocol instance.
        """
        if self.rank == 0:
            if self.args.log_file:
                protocol = Protocol(self.args, self.info_text)
            else:
                protocol = None
        else:
            protocol = None
        return protocol

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

            if self.args.imagenet:
                data_name = 'ImageNet\n'
            elif self.args.train_folder:
                data_name = self.args.train_folder + '\n'
            elif self.args.val_folder:
                data_name = self.args.val_folder + '\n'
            else:
                data_name = 'Random images\n'
            if self.args.distributed:
                global_batch_size = self.args.batch_size * self.args.num_gpus
                local_batch_size = self.args.batch_size
            else:
                global_batch_size = self.args.batch_size
                local_batch_size = int(self.args.batch_size / self.args.num_gpus)

            info_text = f'OS: {platform.uname().system}, {platform.uname().release}\n'\
                        f'Device-name: {platform.uname().node}\n'\
                        f'{self.args.num_gpus} GPU(s) used for benchmark:\n'
            for gpu_id in range(self.args.num_gpus):
                gpu_name = str(torch.cuda.get_device_name(gpu_id))
                info_text += f'{gpu_id}: {gpu_name}\n'
            if not self.args.no_temp:
                try:
                    info_text += f'Nvidia GPU driver version: {GpuInfo.get_driver_version().decode()}\n'
                except NameError:
                    pass
            info_text += f'Available GPUs on device: {torch.cuda.device_count()}\n' \
                         f'Cuda-version: {torch.version.cuda}\n' \
                         f'Cudnn-version: {torch.backends.cudnn.version()}\n' \
                         f'Python-version: {sys.version.split(" ")[0]}\n' \
                         f'PyTorch-version: {torch.__version__}\n' \
                         f'CPU: {cpu_name}\n' \
                         f'Model: {self.args.model}\n' \
                         f'Global batch size: {global_batch_size}\n' \
                         f'Local batch size: {local_batch_size}\n' \
                         f'Distribution Mode: {dist_dict[self.args.distribution_mode]}\n' \
                         f'Process group backend: {self.args.process_group_backend}\n' \
                         f'Optimizer: {optimizer}\n' \
                         f'Precision: {"Automatic mixed precision" if self.args.auto_mixed_precision else self.args.precision}\n' \
                         f'Log file: {self.args.log_file}\n' \
                         f'Training data: {data_name}' \
                         f'Initial learning rate: {self.args.learning_rate}\n' \
                         f'Learning rate decay step: {self.args.step_lr}\n' \
                         f'Used data augmentation: {not self.args.no_augmentation}\n' \
                         f'Checkpoint folder: {self.args.checkpoint_folder}\n' \
                         f'Number of workers: {self.args.num_workers}\n' \
                         f'Warm up steps: {self.args.warm_up_steps}\n' \
                         f'Benchmark start : {start_time}\n'
            print(info_text)
            return info_text
        return None

    def calculate_benchmark(self, rank, epoch, step, total_steps, loss):
        """Calculates the benchmark values and prints status. 
        """
        prompt_str = ''
        duration = self.get_duration()
        if rank == 0:
            self.check_if_warm_up_stage(epoch, step)
            img_per_sec = self.calculate_images_per_sec(epoch, step, duration)
            prompt_str += self.make_benchmark_prompt_string(epoch, step, total_steps, loss, img_per_sec)
            if self.gpu_info:
                gpu_temps = self.gpu_info.get_current_attributes_all_gpus()[0]
                prompt_str += self.gpu_info.get_gpu_info_str()
                self.gpu_temp_list.append(gpu_temps)
            print(prompt_str, end = '')
            if self.args.log_file:
                if self.args.log_benchmark:
                    self.protocol.prompt_str_epoch += prompt_str
        return True

    def finish_epoch(self, rank, epoch, total_steps, loss):
        """Prints infos and writes all epoch plots into the protocol file after each epoch.
        """
        if self.eval_mode:
            self.sum_up_predictions_of_all_gpus()
            val_acc, val_acc5 = self.calculate_validation_accuracy()
        if rank == 0:
            prompt_str = self.make_benchmark_prompt_string(epoch, total_steps, total_steps, loss, None)
            if self.gpu_info:
                gpu_temps = self.gpu_info.get_current_attributes_all_gpus()[0]
                self.gpu_temp_list.append(gpu_temps)
                max_temp_list = self.get_list_of_max_temperatures_of_each_gpu()
                self.gpu_temp_list.clear()
                self.gpu_temp_list.append(max_temp_list)
                prompt_str += self.gpu_info.get_gpu_info_str()
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - self.epoch_start_time
            self.epoch_start_time = epoch_end_time
            epoch_duration_str = Benchmark.make_epoch_duration_string(epoch_duration)
            if not self.eval_mode:
                prompt_str += '\nTraining '
            else:
                prompt_str += f'\nValidation accuracy: {val_acc:.4f}\nValidation accuracy top5: {val_acc5:.4f}\n\nEvaluation '
                self.val_acc_dict[epoch] = val_acc
                self.val_acc5_dict[epoch] = val_acc5
                
            prompt_str += f'epoch finished within {epoch_duration_str}.\n'
            if self.args.mean_img_per_sec:
                prompt_str += self.make_epoch_mean_img_per_sec_string(epoch)
            print(prompt_str)
            
            if self.args.log_file:
                self.protocol.prompt_str_epoch += prompt_str
                self.protocol.update()
                self.protocol.prompt_str_epoch = ''
                if self.eval_mode:
                    self.protocol.update_csv_file(epoch, val_acc, val_acc5)
        return True


    def finish_benchmark(self):
        """Writes benchmark results in a text file and calculates the mean.
        Returns the mean images per sec.
        """
        if self.args.mean_img_per_sec:
            mean_img_per_sec_str = self.make_final_mean_img_per_sec_string()
        else:
            mean_img_per_sec_str = ''
        if self.gpu_temp_list:
            max_temp_str = self.get_max_temperature_str()
        else:
            max_temp_str = ''

        if self.val_acc_dict:
            val_acc_str = f'\nValidation accuracy:\n'
            for key_epoch in self.val_acc_dict:
                val_acc_str += f'    Epoch {key_epoch}: {self.val_acc_dict[key_epoch]:.4f}\n'
        else:
            val_acc_str = ''

        if self.val_acc5_dict:
            val_acc5_str = f'\nValidation accuracy top 5:\n'
            for key_epoch in self.val_acc5_dict:
                val_acc5_str += f'    Epoch {key_epoch}: {self.val_acc5_dict[key_epoch]:.4f}\n'
        else:
            val_acc5_str = ''

        now = datetime.now()
        end_time = now.strftime('%Y/%m/%d %H:%M:%S')
        final_str = mean_img_per_sec_str + max_temp_str + val_acc_str + val_acc5_str + f'\nBenchmark end: {end_time}\n'
        if self.args.log_file:
            self.protocol.finish(final_str)
        print(final_str)

        return True

    def check_if_warm_up_stage(self, epoch, step):
        """Check if the process is still in warm up stage and sets the boolean 
        variable self.warm_up_stage respectively. Returns self.warm_up_stage.
        """
        if step <= self.args.warm_up_steps:
            self.warm_up_stage = True
        else:
            self.warm_up_stage = False
        return self.warm_up_stage

    def cancel_procedure(self):
        """Called in the case of KeyboardInterrupt.
        Finishes the protocol, prints mean and temperature maxima and exits the program.
        """
        if self.args.log_file:
            self.protocol.update()
        self.finish_benchmark()

        if self.args.distributed:
            dist.destroy_process_group()
 
        print('Cancelled by KeyboardInterrupt')
        sys.exit(0)

    def make_benchmark_prompt_string(self, epoch, step, total_steps, loss, img_per_sec):
        """Creates and returns the benchmark prompt string for given arguments.
        """
        prompt_str = f'Epoch [{epoch} / {self.args.load_from_epoch + self.args.num_epochs}], Step [{step} / {total_steps}]'
        try:
            loss_item = loss.detach().item()
            prompt_str += f', Loss: {loss_item:.4f}'
        except AttributeError:
            pass    
        if img_per_sec:
            prompt_str += f', Images per second: {img_per_sec:.1f}'
        prompt_str += '\n'
        return prompt_str

    def calculate_images_per_sec(self, epoch, step, duration):
        """Returns images per second for given arguments.
        """
        img_per_sec = self.args.batch_size * self.args.calc_every * self.args.num_gpus / duration
        if self.args.parallel:
            img_per_sec /= self.args.num_gpus
        if self.args.mean_img_per_sec:
            if not self.warm_up_stage:
                if self.eval_mode:
                    self.img_per_sec_dicts[1][step] = img_per_sec
                else:
                    self.img_per_sec_dicts[0][step] = img_per_sec

        return img_per_sec

    def get_list_of_max_temperatures_of_each_gpu(self):
        """Returns a list containing the maximum temperatures of all gpus ordered by rank.
        """
        gpu_temp_array = np.array(self.gpu_temp_list).transpose()
        return [max(gpu_temp_array[gpu_id]) for gpu_id in range(self.args.num_gpus)]

    def get_max_temperature_str(self):
        """Calculates the maximum temperature for each GPU and returns a string containing these infos.
        """
        max_temp_list = self.get_list_of_max_temperatures_of_each_gpu()
        max_temp_str = f'\nMaximum temperature(s): '
        for gpu_id, max_temp in enumerate(max_temp_list):
            if not gpu_id == 0:
                max_temp_str += '   ||  '
            max_temp_str += f'GPU {gpu_id}: {max_temp} °C'
        return max_temp_str

    def get_duration(self):
        end_time = time.time()
        if self.args.average_duration:
            duration = torch.tensor(end_time - self.start_time).cuda()
            duration = self.average_duration(duration).detach().item()
        else:
            duration = end_time - self.start_time
        self.start_time = end_time
        return duration
    
    def set_to_train_mode(self, model):
        """Set the benchmark and the model to training mode.
        """
        self.eval_mode = False
        model.model.train()
        return model

    def set_to_eval_mode(self, model):
        """Set the benchmark and the model to evaluation mode.
        """
        self.eval_mode = True
        model.model.eval()
        return model

    def sum_up_predictions_of_all_gpus(self):
        """Sums up the correct predictions and the total predictions of the evaluation on each gpu.
        """
        if self.args.distributed:
            dist.barrier()
            dist.reduce(self.correct_predictions, 0, op=dist.ReduceOp.SUM)
            dist.reduce(self.correct_predictions5, 0, op=dist.ReduceOp.SUM)
            dist.reduce(self.total_predictions, 0, op=dist.ReduceOp.SUM)

    def calculate_validation_accuracy(self):
        """Return the validation_accuracy.
        """
        val_acc = self.correct_predictions / self.total_predictions
        val_acc5 = self.correct_predictions5 / self.total_predictions
        self.correct_predictions.zero_()
        self.correct_predictions5.zero_()
        self.total_predictions.zero_()
        return val_acc, val_acc5

    def make_epoch_mean_img_per_sec_string(self, epoch):
        """Calculates the mean images per second for training and/or evaluation for the last epoch and returns it
        in the shape of a printable string.
        """
        mean_img_per_sec_str = f'\nMean images per second in this '
        if not self.eval_mode:
            mean_img_per_sec = self.calc_mean_img_per_sec(self.img_per_sec_dicts[0])
            self.img_per_sec_dicts[0].clear()
            mean_img_per_sec_str += f'training epoch: {mean_img_per_sec}\n'
            self.epoch_mean_img_per_sec_dicts[0][epoch] = mean_img_per_sec
        else:
            mean_img_per_sec = self.calc_mean_img_per_sec(self.img_per_sec_dicts[1])
            self.img_per_sec_dicts[1].clear()
            mean_img_per_sec_str += f'evaluation epoch: {mean_img_per_sec}\n'
            self.epoch_mean_img_per_sec_dicts[1][epoch] = mean_img_per_sec

        return mean_img_per_sec_str

    def make_final_mean_img_per_sec_string(self):
        if self.epoch_mean_img_per_sec_dicts[0]:
            final_mean_img_per_sec_train = self.calc_mean_img_per_sec(self.epoch_mean_img_per_sec_dicts[0])
            final_mean_img_per_sec_eval = self.calc_mean_img_per_sec(self.epoch_mean_img_per_sec_dicts[1])           
        else:
            final_mean_img_per_sec_train = self.calc_mean_img_per_sec(self.img_per_sec_dicts[0])
            final_mean_img_per_sec_eval = self.calc_mean_img_per_sec(self.img_per_sec_dicts[1])

        final_mean_img_per_sec_string = f'\nTotal mean images per second in training: {final_mean_img_per_sec_train}\n' \
                                        f'Total mean images per second in evaluation: {final_mean_img_per_sec_eval}\n'
        return final_mean_img_per_sec_string

    def calc_mean_img_per_sec(self, img_per_sec_dict):
        if len(img_per_sec_dict):
            mean_img_per_sec = round(np.array(list(img_per_sec_dict.values())).mean())
        else:
            mean_img_per_sec = 0
        return mean_img_per_sec

    @staticmethod
    def make_epoch_duration_string(epoch_duration):
        """Makes a duration string splitted in hours/minutes/seconds for
        given duration in seconds.
        """
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

    def average_duration(self, duration):
        """Calculates the average of the duration over all gpus.
        """
        dist.barrier()
        dist.reduce(duration, 0, op=dist.ReduceOp.SUM)
        duration /= self.args.num_gpus
        return duration


class Protocol(object):
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

    def update(self):
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
    def update_csv_file(self, epoch, val_acc, val_acc5):
        with open(self.log_file.with_suffix('.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((epoch, val_acc.item(), val_acc5.item()))

    def finish(self, final_str):
        """Writes given string with benchmark end results like mean images per second, maximum GPU temperatures 
        and the validation accuracy from each epoch into the protocol file.
        """
        with open(self.log_file, 'a') as log_file:
            log_file.write(final_str)
        return True


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
