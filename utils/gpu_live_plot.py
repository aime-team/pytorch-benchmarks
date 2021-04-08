import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nvidia_smi
import time
import argparse
import sys

#matplotlib.use('Qt5Agg')


def get_temperature_and_update_axes(
        i, plt, num_gpus, start, time_list, gpu_temp_list, fan_speed_list, axis_gpu_temp_list, axis_fan_speed_list
                                    ):
    plt.tight_layout(w_pad=1, h_pad=1)

    for gpu_id in range(num_gpus):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
        gpu_temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
        fan_speed = nvidia_smi.nvmlDeviceGetFanSpeed(handle)
        now = time.time()
        time_list[gpu_id].append(now - start)
        gpu_temp_list[gpu_id].append(int(gpu_temp))
        fan_speed_list[gpu_id].append(int(fan_speed))
        axis_gpu_temp_list[gpu_id].clear()
        axis_fan_speed_list[gpu_id].clear()
        axis_gpu_temp_list[gpu_id].set_xlabel('Time in seconds')
        axis_gpu_temp_list[gpu_id].set_ylabel('Temperature in °C', color='tab:red')
        axis_gpu_temp_list[gpu_id].set_title(f'GPU {gpu_id}')
        axis_fan_speed_list[gpu_id].set_ylabel('Fan Speed in %', color='tab:blue')
        axis_fan_speed_list[gpu_id].set_ylim(0, 100)
        axis_gpu_temp_list[gpu_id].yaxis.set_major_locator(MaxNLocator(integer=True))
        axis_fan_speed_list[gpu_id].yaxis.set_major_locator(MaxNLocator(integer=True))
        axis_gpu_temp_list[gpu_id].grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
        axis_gpu_temp_list[gpu_id].plot(time_list[gpu_id], gpu_temp_list[gpu_id], color='tab:red')
        axis_fan_speed_list[gpu_id].plot(time_list[gpu_id], fan_speed_list[gpu_id], color='tab:blue')
        plt.savefig('temp_fan_speed_log.png')


def make_live_plot_gpu_temp_and_fan_speed(num_gpus, refresh_interval):
    fig = plt.figure(figsize=(6, 4*num_gpus))
    axis_gpu_temp_list = []
    axis_fan_speed_list = []

    for gpu_id in range(num_gpus):
        axis_gpu_temp = fig.add_subplot(num_gpus, 1, gpu_id + 1)
        axis_gpu_temp.set_title(f'GPU {gpu_id}')
        axis_fan_speed = axis_gpu_temp.twinx()
        axis_gpu_temp.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis_fan_speed.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis_gpu_temp_list.append(axis_gpu_temp)
        axis_fan_speed_list.append(axis_fan_speed)
        axis_gpu_temp.set_xlabel('Time in seconds')
        axis_gpu_temp.set_ylabel('Temperature in °C', color='tab:red')
        axis_fan_speed.set_ylabel('Fan Speed in %', color='tab:blue')
        axis_fan_speed.set_ylim(0, 100)
    time_list = [[] for gpu_id in range(num_gpus)]
    gpu_temp_list = [[] for gpu_id in range(num_gpus)]
    fan_speed_list = [[] for gpu_id in range(num_gpus)]
    plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
    plt.tight_layout(w_pad=1, h_pad=1)
    start = time.time()
    _ = matplotlib.animation.FuncAnimation(
        fig, get_temperature_and_update_axes, interval=refresh_interval,
        fargs=(plt, num_gpus, start, time_list, gpu_temp_list, fan_speed_list, axis_gpu_temp_list, axis_fan_speed_list)
                                           )
    plt.show(block=False)


def main():
    """
    parser = argparse.ArgumentParser(
        description='GPU temp and fan speed live plot', formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument(
        '-ng', '--num_gpus', type=int, default=1, required=False,
        help='Number of gpus.'
                        )
    parser.add_argument(
        '-ri', '--refresh_interval', type=int, default=500, required=False,
        help='Change live plot refresh interval in ms.'
                        )
    args = parser.parse_args()"""
    print("Check2")
    nvidia_smi.nvmlInit()
    make_live_plot_gpu_temp_and_fan_speed(2, 500)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("plot interrupt")
