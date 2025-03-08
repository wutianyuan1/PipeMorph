import os
import re
import numpy as np
import matplotlib.pyplot as plt

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/nic'
MODEL = '14B'
METHODS = ['ZB', 'ZB-CPU']
NUM_ITERS = 300
BATCH_SIZE = 24
COLORS = ['#1f77b4', '#ff7f0e']

plt.figure(figsize=(14, 2.5))
method_times= []
for method in METHODS:
    x = np.arange(NUM_ITERS)
    dir = os.path.join(PATH, MODEL, method)
    try:
        with open(f"{dir}/log.txt", 'r') as f:
            iters = f.readlines()
            assert len(iters) == NUM_ITERS
            iters = [iter.split(' | ')[2] for iter in iters]
            iter_times = np.array([float(re.findall(pattern, iter)[0]) for iter in iters]) / 1000.0
            if method == 'ZB-CPU':
                label = 'PipeMorph-CPU'
            else:
                label = method
            throughput_per_iter = BATCH_SIZE / iter_times
            t = np.cumsum(iter_times)
            color = COLORS[METHODS.index(method)]
            plt.hlines(throughput_per_iter[0], 0, t[0], color, label=label)
            for i in range(1, NUM_ITERS):
                plt.hlines(throughput_per_iter[i], t[i - 1], t[i], color)
            for i in range(NUM_ITERS - 1):
                ymin = min(throughput_per_iter[i], throughput_per_iter[i + 1])
                ymax = max(throughput_per_iter[i], throughput_per_iter[i + 1])
                plt.vlines(t[i], ymin, ymax, color)
            plt.hlines(BATCH_SIZE * NUM_ITERS / np.sum(iter_times), 0, t[-1], color=color, linestyle='--')
    except:
        continue

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Throughput (/s)', fontsize=14)
plt.grid(linestyle='-.')
legend = plt.legend(loc='lower right', fontsize=12)
plt.savefig(f'{PATH}/{MODEL}.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.savefig(f'{PATH}/{MODEL}.png', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
