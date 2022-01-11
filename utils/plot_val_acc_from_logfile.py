import re
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib.widgets import CheckButtons


def load_flags():
    parser = argparse.ArgumentParser(
        description='PyTorch Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )

    parser.add_argument(
        '-lf', '--log_folder', type=str, required=False,
        help='Destination of the folder containing the log files to load.'
                        )
    parser.add_argument(
        '-sl', '--specific_log_files', type=str, nargs='+', required=False,
        help='Loads given log files instead of loading all files from --log_folder. Takes any number of log files'
                        )
    parser.add_argument(
        '-t', '--title', type=str, required=False,
        help='Set the title of the diagram.'
                        )
    args = parser.parse_args()
    return args


def get_log_file_list(args):
    log_file_list = []
    if args.specific_log_files:
        log_file_list = [Path(log_file) for log_file in args.specific_log_files]
    elif args.log_folder:
        for log_file in Path(args.log_folder).glob('*.txt'):
            log_file_list.append(log_file)
    else:
        log_file_folder = Path.cwd().parent / 'log'
        for log_file in log_file_folder.glob('*'):
            log_file_list.append(log_file)
    log_file_list.sort()
    return log_file_list


def read_log_file(log_file):
    pattern = re.compile(r"Epoch (\d+): (\S+)")
    x_axes_epoch = []
    y_axes_val_acc = []
    with open(log_file, 'r') as logfile:
        log = logfile.read()
        for match in re.finditer(pattern, log):
            x_axes_epoch.append(int(match.group(1)))
            y_axes_val_acc.append(float(match.group(2)))
    return x_axes_epoch, y_axes_val_acc


def on_pick(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    legline = event.artist
    origline = legend_dict[legline]
    visible = not origline.get_visible()
    origline.set_visible(visible)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled.
    legline.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()


def make_diagram(log_file_list, ax):
    lines = []
    print('Used log files:')
    for log_file in log_file_list:
        print(log_file)
        x_axes, y_axes = read_log_file(log_file)
        label = log_file.stem
        line, = ax.plot(x_axes, y_axes, label=label)
        lines.append(line)
    legend_dict = {}
    leg = ax.legend(fancybox=True, shadow=True)
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)  # Enable picking on the legend line.
        legend_dict[legline] = origline
    return legend_dict


if __name__ == '__main__':
    args = load_flags()
    fig, ax = plt.subplots()
    log_file_list = get_log_file_list(args)
    legend_dict = make_diagram(log_file_list, ax)
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.title(args.title)
    plt.xlabel('Training epoch')
    plt.ylabel('Validation accuracy')
    plt.show()
