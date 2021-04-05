#!/usr/bin/env python3

import torch
import platform
import sys
import subprocess
import datetime
import time
import torch.distributed as dist
import numpy as np
import subprocess
import threading


try:
    import nvidia_smi
    nvidia_smi.nvmlInit()

    class GpuInfo(object):
        """Collects GPU information using nvidia-ml-py3.
        """

        def __init__(self, num_gpus):
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
                                f'Memory used: {round((self.memory_usage_list[gpu_id][0] / 1024) / 1024, 2)} / ' \
                                f'{round((self.memory_usage_list[gpu_id][1] / 1024) / 1024, 2)} MB\n'
            return gpu_info_str

except ModuleNotFoundError:
    pass


def run_live_plot_thread(num_gpus, refresh_interval=500):
    """ Runs the python-script gpu_liveplot.py via subprocess.
    """
    thread = threading.Thread(
        target=subprocess.run, args=(
            ['python3', 'gpu_live_plot.py', '--num_gpus', str(num_gpus), '--refresh_interval', str(refresh_interval)],
                                     )
                              )
    thread.start()
    return thread


def make_info_text(args, start_time):
    """Makes info text about the device and the OS shown at the beginning of the benchmark and in the protocol.
    """
    info_text = f'OS: {platform.uname().system}, {platform.uname().release}\n'\
                 f'Device-name: {platform.uname().node}\n'\
                 f'{args.num_gpus} GPU(s) used for benchmark:\n'
    for gpu_id in range(args.num_gpus):
        gpu_name = str(torch.cuda.get_device_name(gpu_id))
        info_text += f'{gpu_id}: {gpu_name}\n'
    try:
        info_text += f'Nvidia GPU driver version: {GpuInfo.get_driver_version().decode()}\n'
    except NameError:
        pass
    info_text += f'Available GPUs on device: {torch.cuda.device_count()}\n' \
                 f'Cuda-version: {torch.version.cuda}\n' \
                 f'Cudnn-version: {torch.backends.cudnn.version()}\n' \
                 f'Python-version: {sys.version.split(" ")[0]}\n' \
                 f'PyTorch-version: {torch.__version__}\n'

    cpu_name = 'unknown'
    for line in subprocess.check_output("lscpu", shell=True).strip().decode().split('\n'):
        if 'Modellname' in line or 'Model name' in line:
            cpu_name = line.split(':')[1].strip()

    info_text += f'CPU: {cpu_name}\n' \
                 f'Model: {args.model}\n' \
                 f'Batch size: {args.batch_size}\n' \
                 f'Precision: {args.precision}\n'

    if args.img_folder:
        info_text += f'Training data: {args.img_folder}\n'
    else:
        info_text += f'Training data: Random images\n'

    info_text += f'Warm up steps: {args.warm_up_steps}\n'
    info_text += f'Benchmark start : {start_time}\n\n'
    return info_text


def make_protocol(img_per_sec_dict, gpu_info_dict, gpu_temp_list, args, start_time):
    """Writes benchmark results in a textfile and calculates the mean.
    Takes the images_per_sec_dict and the infotext as arguments. Returns the mean of images per sec.
    """
    info_text = make_info_text(args, start_time)
    if len(img_per_sec_dict):
        mean_img_per_sec = round(np.array(list(img_per_sec_dict.values())).mean(), 2)
    else:
        mean_img_per_sec = 0
    mean_img_per_sec_str = f'\nMean images per second: {mean_img_per_sec}'
    if gpu_temp_list:
        gpu_temp_array = np.array(gpu_temp_list).transpose()
        max_temp_list = [max(gpu_temp_array[gpu_id]) for gpu_id in range(args.num_gpus)]
        max_temp_str = f'\nMaximum temperature(s): '
        for gpu_id, max_temp in enumerate(max_temp_list):
            if not gpu_id == 0:
                max_temp_str += '   ||  '
            max_temp_str += f'GPU {gpu_id}: {max_temp} °C'
    else:
        max_temp_str = ''

    now = datetime.datetime.now()
    end_time = now.strftime('%Y/%m/%d %H:%M:%S')
    with open(f'{args.num_gpus}_{str(torch.cuda.get_device_name(0))}_{args.batch_size}.txt', 'w') as protocol:
        protocol.write(info_text)
        for key_img_per_sec in img_per_sec_dict:
            protocol.write(
                f'Epoch: {key_img_per_sec[0]}, Step: {key_img_per_sec[1]}, '
                f'Images per second: {img_per_sec_dict[key_img_per_sec]}\n'
                           )
        if gpu_info_dict:
            for key_gpu_info in gpu_info_dict:
                protocol.write(gpu_info_dict[key_gpu_info])
        protocol.write(mean_img_per_sec_str)

        protocol.write(max_temp_str)
        protocol.write(f'\n\nBenchmark end : {end_time}\n')
    return mean_img_per_sec_str, max_temp_str


def cancel_procedure(epoch, step, args, img_per_sec_dict, gpu_info_dict, gpu_temp_list, start_time):
    """Makes protocol and prints mean and temperature maxima, if KeyboardInterrupt.
    """
    if epoch == args.start_epoch and step < args.warm_up_steps:
        if not args.parallel:
            dist.destroy_process_group()
        sys.exit('Cancelled in warm up stage')
    else:
        if len(img_per_sec_dict):
            print(*make_protocol(img_per_sec_dict, gpu_info_dict, gpu_temp_list, args, start_time))

        if not args.parallel:
            dist.destroy_process_group()
        sys.exit(f'\nCancelled by KeyboardInterrupt')


def make_benchmark_calculations(
        start, epoch, step, total_step, correct_predictions, total_predictions, loss,
        args, gpu_info, rank, img_per_sec_dict, gpu_temp_list, gpu_info_dict
                                ):
    """Calculates the benchmark and prints status.
    """
    if (step + 1) % args.calc_every == 0:
        end = time.time()
        duration = torch.tensor(end - start).cuda()
        start = end
        if not args.parallel:
            duration = average_duration(duration, args.num_gpus)
        if rank == 0:
            img_per_sec = args.batch_size * args.calc_every * args.num_gpus / duration.item()
            try:
                loss_item = loss.item()
            except AttributeError:
                loss_item = 0.0
            output_str = \
                f'Epoch [{epoch} / {args.start_epoch + args.num_epochs - 1}], Step [{(step + 1)} / {total_step}], '
            if args.eval and total_predictions:
                val_acc = correct_predictions / total_predictions
                output_str += f'Validation accuracy: {val_acc:.04}, '
            else:
                output_str += f'Loss: {loss_item:.4f}, '
            output_str += f'Images per second: {img_per_sec}'
            print(output_str)

            if gpu_info:
                gpu_temps = gpu_info.get_current_attributes_all_gpus()[0]
                gpu_info_str = gpu_info.get_gpu_info_str()
                print(gpu_info_str)
                if step > args.warm_up_steps:
                    gpu_temp_list.append(gpu_temps)
                    gpu_info_dict[(epoch, step + 1)] = gpu_info_str
            if step > args.warm_up_steps:
                img_per_sec_dict[(epoch, step + 1)] = img_per_sec
    if (step + 1) == total_step:
        try:
            loss_item = loss.item()
        except AttributeError:
            loss_item = 0.0
        print(
            f'Epoch [{epoch} / {args.start_epoch + args.num_epochs - 1}], '
            f'Step [{(step + 1)} / {total_step}], Loss: {loss_item:.4f}\n\n'
              )
        start = time.time()
    return start


def average_duration(duration , num_gpus):
    """Calculates the average of the duration over all gpus.
    """
    dist.all_reduce(duration, op=dist.ReduceOp.SUM)
    duration /= num_gpus
    return duration


def average_gradients(model, num_gpus):
    """Calculates the average of the gradients over all gpus.
    """
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= num_gpus

