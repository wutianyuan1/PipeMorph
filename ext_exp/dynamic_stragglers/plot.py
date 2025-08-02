import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ae.parse_time import parse_iter_times

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'ext_exp/dynamic_stragglers'
MODELS = ['7B']
METHODS = ['1F1B', 'ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
HATCHES = ['//', '--', '\\\\', '||']
COUNT_LAST_ITERS = 5
WIDTH = 0.23
# COLORS = ['#E1D5E7', '#F8CECC', '#FFE6CC', '#DAE8FC']
COLORS = ['#9673A6', '#E7AB10', '#6C8EBF', '#B85450']
COLORS2 = ['#462356', '#975B00', '#1C3E6F', '#780400']

opt_ratio = {method: None for method in METHODS}
t = {method: None for method in METHODS}

plt.figure(figsize=(14, 3.6))
x = np.arange(len(MODELS))
method_times = []
for method in METHODS:
    times = []
    std_times = []
    for i, model in enumerate(MODELS):
        dir = os.path.join(PATH, method)
        try:
            assert os.path.exists(f"{dir}/real.png")
        except:
            print(f"{dir}/real.png does not exist")
        try:
                iter_times = parse_iter_times(dir, method, (-5, None))
                iter_times /= 1000
                avg_iter_time = np.mean(iter_times)
                std_iter_time = np.std(iter_times)
                # print(slowlink, delay, model, method, avg_iter_time, std_iter_time)
                times.append(avg_iter_time)
                std_times.append(std_iter_time)
        except:
                times.append(0)
                std_times.append(0)
        plt.text(x[i], times[-1] + 1.5*std_times[-1], "{:.2f}".format(times[-1]), va='bottom', ha='center', fontdict={"fontsize": 12})
    if method == 'ZB-CPU':
        label = 'PipeMorph-CPU'
    elif method == 'ZB-CPU-ReSchedule':
        label = 'PipeMorph'
    else:
        label = method
    plt.bar(x, times, width=WIDTH, label=label, zorder=100, alpha=0.7, edgecolor=COLORS2[METHODS.index(method)], hatch=HATCHES[METHODS.index(method)], color=COLORS[METHODS.index(method)])
    plt.errorbar(x, times, yerr=std_times, color='black', fmt='o', ecolor='black', capsize=5, zorder=100)
    x = x + WIDTH
    method_times.append(times)
method_times = np.array(method_times)
# print(f"{slowlink}, {delay}: Optimize ratio: {method_times[1] / method_times[3]}")  # ZB / ours
for i, method in enumerate(METHODS):
    r = 1 - method_times[3] / method_times[i]
    print(f"1 - PipeMorph / {method} = {r}")
    opt_ratio[method] = r
    t[method] = method_times[i]
x = np.arange(len(MODELS)) + (len(METHODS) - 1) / 2 * WIDTH
plt.xticks(x, MODELS, fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Time Per\nIteration (s)', fontsize=12)
plt.grid(linestyle='-.')
plt.tight_layout(pad=1.05)
legend = plt.figlegend(loc=(0.27, 0.9), ncols=4, fontsize=14, frameon=False)
plt.savefig(f'{PATH}/dynamic_stragglers.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.savefig(f'{PATH}/dynamic_stragglers.png', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
for method in opt_ratio:
    print(method)
    for i, model in enumerate(MODELS):
        vals = [opt_ratio[method][i]]
        print(model, vals)
        print(f'avg {sum(vals) / len(vals):.3f}')
        print(f'max {max(vals):.3f}')
        print()