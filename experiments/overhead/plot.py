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
WIDTH = 0.25
COLORS = ['#9673A6', '#E7AB10', '#6C8EBF', '#B85450']
COLORS2 = ['#462356', '#975B00', '#1C3E6F', '#780400']

plt.figure(figsize=(7, 2))
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
                iter_times = np.array([float(re.findall(pattern, iter)[0]) for iter in iters])/1000.0
                avg_iter_time = np.mean(iter_times)
                times.append(avg_iter_time)
                std_times.append(np.std(iter_times))
        except:
                times.append(0)
                std_times.append(0)
        plt.text(x[i], times[-1] + 2 * std_times[-1], "{:.2f}".format(times[-1]), va='bottom', ha='center', fontsize=12, zorder=100)
    plt.bar(x, times, width=WIDTH, label=method if method != 'ALL-CPU' else 'PipeMorph-All-CPU', alpha=0.7, zorder=100, edgecolor=COLORS2[METHODS.index(method)], hatch=HATCHES[METHODS.index(method)], color=COLORS[METHODS.index(method)])
    plt.errorbar(x, times, yerr=std_times, color='black', fmt='o', ecolor='black', capsize=5, zorder=100)
    x = x + WIDTH
    method_times.append(times)
method_times = np.array(method_times)
method_times_dict = {method: method_times[i].tolist() for (i, method) in enumerate(METHODS)}
method_times_dict['ZB-CPU'] = method_times_dict['ALL-CPU']
method_times_dict['ZB-CPU-ReSchedule'] = method_times_dict['ALL-CPU']

x = np.arange(len(MODELS)) + (len(METHODS) - 1) / 2 * WIDTH
plt.ylim(0, 1.4)
plt.xticks(x, MODELS, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Time Per\nIteration (s)', fontsize=12)
plt.grid(linestyle='-.')
legend = plt.legend(loc=(0.07, 0.95), ncols=3, fontsize=12, frameon=False)
plt.savefig(f'{ALL_CPU_PATH}/overhead.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
print(method_times_dict, file=open(f'{ALL_CPU_PATH}/times.txt', 'w'))
plt.close()
