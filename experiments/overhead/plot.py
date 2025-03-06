import os
import re
import numpy as np
import matplotlib.pyplot as plt

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/single'
ALL_CPU_PATH = 'experiments/overhead'
MODELS = ['7B', '14B', '30B', '60B']
METHODS = ['1F1B', 'ZB', 'ALL-CPU']
DELAYS = ['0.03', '0.06']
HATCHES = ['//', '--', '\\\\', '||']
WIDTH = 0.2

plt.figure(figsize=(7, 2.5))
method_times = []
x = np.arange(len(MODELS))
for method in METHODS:
    times = []
    std_times = []
    for i, model in enumerate(MODELS):
        if method != 'ALL-CPU':
            dir = os.path.join(PATH, model, method, '0_1', '0.03')  # choose the beginning iters of 0_1/0.03
        else:
            dir = os.path.join(ALL_CPU_PATH, model, method)

        try:
            with open(f"{dir}/log.txt", 'r') as f:
                iters = f.readlines()[2:7]
                iters = [iter.split(' | ')[2] for iter in iters]
                iter_times = np.array([float(re.findall(pattern, iter)[0]) for iter in iters])
                avg_iter_time = np.mean(iter_times)
                times.append(avg_iter_time)
                std_times.append(np.std(iter_times))
        except:
                times.append(0)
                std_times.append(0)
        plt.text(x[i], times[-1] + 1.5 * std_times[-1], round(times[-1]), va='bottom', ha='center')
    plt.bar(x, times, width=WIDTH, label=method, zorder=100, edgecolor='black', hatch=HATCHES[METHODS.index(method)])
    plt.errorbar(x, times, yerr=std_times, color='black', fmt='o', ecolor='black', capsize=5, zorder=100)
    x = x + WIDTH
    method_times.append(times)
method_times = np.array(method_times)
method_times_dict = {method: method_times[i].tolist() for (i, method) in enumerate(METHODS)}
method_times_dict['ZB-CPU'] = method_times_dict['ALL-CPU']
method_times_dict['ZB-CPU-ReSchedule'] = method_times_dict['ALL-CPU']

x = np.arange(len(MODELS)) + (len(METHODS) - 1) / 2 * WIDTH
plt.xticks(x, MODELS, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Time Per\nIteration (ms)', fontsize=12)
plt.grid(linestyle='-.')
legend = plt.legend(fontsize=12)
plt.savefig(f'{ALL_CPU_PATH}/overhead.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
print(method_times_dict, file=open(f'{ALL_CPU_PATH}/times.txt', 'w'))
plt.close()
