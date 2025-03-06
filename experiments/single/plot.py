import os
import re
import numpy as np
import matplotlib.pyplot as plt

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/single'
MODELS = ['7B', '14B', '30B', '60B']
METHODS = ['1F1B', 'ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
DELAYS = ['0.03', '0.06']
HATCHES = ['//', '--', '\\\\', '||']
COUNT_LAST_ITERS = 5
WIDTH = 0.2

for slowlink in ['First', 'Last']:
    plt.figure(figsize=(14, 2))
    for figid, delay in enumerate(DELAYS):
        x = np.arange(len(MODELS))
        plt.subplot(1, 2, figid + 1)
        method_times = []
        for method in METHODS:
            times = []
            std_times = []
            for i, model in enumerate(MODELS):
                dir = os.path.join(PATH, model, method, '0_1' if slowlink == 'First' else ('2_3' if model in ['7B'] else '6_7'), delay)
                try:
                    assert os.path.exists(f"{dir}/real.png")
                except:
                    print(f"{slowlink, delay, model, method}")
                try:
                    with open(f"{dir}/log.txt", 'r') as f:
                        iters = f.readlines()[-COUNT_LAST_ITERS :]
                        iters = [iter.split(' | ')[2] for iter in iters]
                        iter_times = np.array([float(re.findall(pattern, iter)[0]) for iter in iters])
                        avg_iter_time = np.mean(iter_times)
                        std_iter_time = np.std(iter_times)
                        # print(slowlink, delay, model, method, avg_iter_time, std_iter_time)
                        times.append(avg_iter_time)
                        std_times.append(std_iter_time)
                except:
                        times.append(0)
                        std_times.append(0)
                plt.text(x[i], times[-1] + 1.5*std_times[-1], round(times[-1]), va='bottom', ha='center')
            if method == 'ZB-CPU':
                label = 'PipeMorph-CPU'
            elif method == 'ZB-CPU-ReSchedule':
                label = 'PipeMorph'
            else:
                label = method
            plt.bar(x, times, width=WIDTH, label=label if figid == 0 else None, zorder=100, edgecolor='black', hatch=HATCHES[METHODS.index(method)])
            plt.errorbar(x, times, yerr=std_times, color='black', fmt='o', ecolor='black', capsize=5, zorder=100)
            x = x + WIDTH
            method_times.append(times)
        method_times = np.array(method_times)
        print(f"{slowlink}, {delay}: Optimize ratio: {method_times[1] / method_times[3]}")  # ZB / ours
        x = np.arange(len(MODELS)) + (len(METHODS) - 1) / 2 * WIDTH
        plt.xticks(x, MODELS, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Model', fontsize=14)
        if figid == 0:
            plt.ylabel('Time Per\nIteration (ms)', fontsize=12)
        # plt.title(f'{float(delay) * 1000}ms Delay on {slowlink} Link')
        plt.grid(linestyle='-.')
    legend = plt.figlegend(loc=(0.3, 0.88), ncols=4, fontsize=12)
    plt.savefig(f'{PATH}/{slowlink}.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
    plt.savefig(f'{PATH}/{slowlink}.png', bbox_inches='tight', bbox_extra_artists=(legend,))
    plt.close()
