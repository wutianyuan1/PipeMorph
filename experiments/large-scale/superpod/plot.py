import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/large-scale/superpod'
METHODS = ['1F1B', 'ZB', 'ZB-CPU-ReSchedule']
WARMUP_ITERS = 2
TOTAL_ITERS = 1200
NUM_ITERS = TOTAL_ITERS - WARMUP_ITERS
BATCH_SIZE = 8 * 3 * 2
# COLORS = ['#1f77b4', '#ff7f0e', 'red']
LINESTYLES = ['solid', 'solid', 'solid']
MAX_T = 3000
COLORS = ['#9673A6', '#D79B00', '#B85450']

plt.figure(figsize=(14, 7.5))
method_throughput = []
method_times= []
for method in METHODS:
    x = np.arange(NUM_ITERS)
    dir = os.path.join(PATH, method)
    plt.subplot(3, 1, METHODS.index(method) + 1)
    try:
        with open(f"{dir}/log.txt", 'r') as f:
            iters = f.readlines()
            # assert len(iters) == NUM_ITERS + WARMUP_ITERS
            iters = [iter.split(' | ')[2] for iter in iters]
            iter_times = np.array([float(re.findall(pattern, iter)[0]) for iter in iters])[WARMUP_ITERS:TOTAL_ITERS] / 1000.0
            if method == 'ZB-CPU-ReSchedule':
                label = 'PipeMorph'
            else:
                label = method
            throughput_per_iter = BATCH_SIZE / iter_times
            t = np.cumsum(iter_times)
            ls = LINESTYLES[METHODS.index(method)]
            color = COLORS[METHODS.index(method)]
            # plt.plot(np.arange(len(iter_times)), savgol_filter(throughput_per_iter, 5, 3), linestyle=ls, color=color, label=label)
            print(method, np.mean(throughput_per_iter), BATCH_SIZE * NUM_ITERS / np.sum(iter_times))
            method_throughput.append(BATCH_SIZE * NUM_ITERS / np.sum(iter_times))
            method_times.append(np.sum(iter_times))
            # throughput_per_iter = savgol_filter(throughput_per_iter, 7, 3)

            plt.grid(axis='y', linestyle='-.', color='#d3d3d3')
            for x in range(0, MAX_T, 250):
                plt.axvline(x=x, linestyle='-.', color='#d3d3d3', linewidth=3)

            plt.hlines(throughput_per_iter[0], 0, t[0], color, ls, label=label)
            for i in range(1, NUM_ITERS):
                plt.hlines(throughput_per_iter[i], t[i - 1], t[i], color, ls, linewidth=3)
            for i in range(NUM_ITERS - 1):
                ymin = min(throughput_per_iter[i], throughput_per_iter[i + 1])
                ymax = max(throughput_per_iter[i], throughput_per_iter[i + 1])
                plt.vlines(t[i], ymin, ymax, color, ls, linewidth=3)
            
            plt.hlines(BATCH_SIZE * NUM_ITERS / np.sum(iter_times), 0, MAX_T, color='black', linestyle='--', linewidth=3)
            throughput = BATCH_SIZE * NUM_ITERS / np.sum(iter_times)
            plt.text(MAX_T, throughput, "{:.2f}".format(throughput), va='bottom', ha='right', fontdict={"fontsize": 22})
            plt.xlim(0, MAX_T)
            plt.yticks(fontsize=22)
            if METHODS.index(method) == 1:
                plt.ylabel('Throughput (/s)', fontsize=22)
            if METHODS.index(method) != 2:
                plt.xticks([])
            else:
                plt.xticks(fontsize=22)
                plt.xlabel('Wallclock Time (s)', fontsize=22)
    except:
        continue
legend = plt.figlegend(loc=(0.25, 0.92), ncols=3, fontsize=20)
plt.savefig(f'{PATH}/large-scale.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.savefig(f'{PATH}/large-scale.png', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
for i in range(len(METHODS)):
    print(METHODS[i], f'{method_times[i] / 60:.1f} min', f'{method_throughput[-1] / method_throughput[i]:.3f}')