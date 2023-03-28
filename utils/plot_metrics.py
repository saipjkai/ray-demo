from argparse import ArgumentParser
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    pass


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--output_dir", default="plots", help="Path to store plots")
    ap.add_argument("--input_dir", default="", help="Path to instance metrics")


if __name__ == "__main__":
    main()

t3a2xlarge_instances = ('w/o ray', '1H', '1H + 1W', '1H + 2W')
t3a2xlarge_results = {
    'Inference Time(sec)': (478.17, 137.67, 80.96, 57.457),
    'FPS': (5.01, 17.42, 29.62, 41.73),
    'FPS/Instance': (5.01, 17.42, 14.81, 13.90),
    'FPS/1core': (0.63, 2.18, 1.85, 1.74),
}

# m5a4xlarge_instances = ('w/o ray', '1H', '1H + 1W')
# m5a4xlarge_results = {
#     'Inference Time': (464.70, 77.57, 45.86),
#     'FPS': (5.16, 30.91, 52.29),
#     'FPS/Instance': (5.16, 30.91, 26.14),
#     'FPS/1core': (0.32, 1.93, 1.63),
# }

x = np.arange(len(t3a2xlarge_instances))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(constrained_layout=True)

for attribute, measurement in t3a2xlarge_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Results')
ax.set_title('YOLOv8 inference on longdemo.mp4')
# ax.set_title('Instance: m5a4xlarge & I/P: longdemo.mp4 (38 sec\'s @ 60 FPS ~ 2398 frames)')
ax.set_xticks(x + width, t3a2xlarge_instances)
ax.legend(loc='upper left', ncols=4)
ax.set_ylim(0, 500)

plt.savefig('metrics.png')
plt.show()