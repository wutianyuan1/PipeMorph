import os
import re
import numpy as np
import matplotlib.pyplot as plt

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/single'
MODELS = ['7B', '14B', '30B']
METHODS = ['ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
DELAYS = ['0.03', '0.06']
COUNT_LAST_ITERS = 5
WIDTH = 0.2

for slowlink in ['First', 'Last']:
    for delay in DELAYS:
        x = np.arange(len(MODELS))
        for method in METHODS:
            times = []
            for i, model in enumerate(MODELS):
                dir = os.path.join(PATH, model, method, '0_1' if slowlink == 'First' else ('2_3' if model == '7B' else '6_7'), delay)
                try:
                    assert os.path.exists(f"{dir}/real.png")
                except:
                    print(f"{slowlink, delay, model, method}")
                with open(f"{dir}/log.txt", 'r') as f:
                    iters = f.readlines()[-COUNT_LAST_ITERS :]
                    iters = [iter.split(' | ')[2] for iter in iters]
                    iter_times = [float(re.findall(pattern, iter)[0]) for iter in iters]
                    avg_iter_time = sum(iter_times) / len(iter_times)
                    print(slowlink, delay, model, method, avg_iter_time)
                    times.append(avg_iter_time)
                plt.text(x[i], times[-1], round(times[-1]), va='bottom', ha='center')
            plt.bar(x, times, width=WIDTH, label=method)
            x = x + WIDTH
        x = np.arange(len(MODELS)) + (len(MODELS) - 1) / 2 * WIDTH
        plt.xticks(x, MODELS)
        plt.xlabel('Model')
        plt.ylabel('Time Per Iteration (ms)')
        plt.legend()
        plt.title(f'{float(delay) * 1000}ms Delay on {slowlink} Link')
        plt.savefig(f'{PATH}/{slowlink}_{delay}.png')
        plt.close()