import os
import re
import numpy as np
import matplotlib.pyplot as plt

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/seqlen'
MODEL = '14B'
METHODS = ['1F1B', 'ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
SEQLENS=["512", "1024", "2048", "4096", "8192"]
HATCHES = ['//', '--', '\\\\', '||']
COUNT_LAST_ITERS = 5
WIDTH = 0.23
DELAY = "0.03"
COLORS = ['#9673A6', '#E7AB10', '#6C8EBF', '#B85450']
COLORS2 = ['#462356', '#975B00', '#1C3E6F', '#780400']

opt_ratio = {method: {seq_len: 0 for seq_len in SEQLENS} for method in METHODS}

plt.figure(figsize=(7, 2))
method_times = []
x = np.arange(len(SEQLENS))
for method in METHODS:
    times = []
    std_times = []
    for i, seq_len in enumerate(SEQLENS):
        dir = os.path.join(PATH, MODEL, method, seq_len, DELAY)
        try:
            assert os.path.exists(f"{dir}/real.png")
        except:
            print(f"{seq_len, DELAY, MODEL, method}")
        try:
            with open(f"{dir}/log.txt", 'r') as f:
                iters = f.readlines()[-COUNT_LAST_ITERS :]
                iters = [iter.split(' | ')[2] for iter in iters]
                iter_times = np.array([float(re.findall(pattern, iter)[0]) for iter in iters])/1000.0
                avg_iter_time = np.mean(iter_times)
                std_iter_time = np.std(iter_times)
                times.append(avg_iter_time)
                std_times.append(std_iter_time)
        except:
                times.append(0)
                std_times.append(0)
        plt.text(x[i], times[-1] + 1.5*std_times[-1], "{:.2f}".format(times[-1]), fontdict={"fontsize": 12}, va='bottom', ha='center', zorder=101)
    if method == 'ZB-CPU':
        label = 'PipeMorph-CPU'
    elif method == 'ZB-CPU-ReSchedule':
        label = 'PipeMorph'
    else:
        label = method
    plt.bar(x, times, width=WIDTH, label=label, zorder=100, hatch=HATCHES[METHODS.index(method)], color=COLORS[METHODS.index(method)], edgecolor=COLORS2[METHODS.index(method)], alpha=0.7)
    plt.errorbar(x, times, yerr=std_times, color='black', fmt='o', ecolor='black', capsize=5, zorder=100)
    x = x + WIDTH
    method_times.append(np.array(times))
for i, method in enumerate(METHODS):
    r = 1 - method_times[3] / method_times[i]
    print(f"1 - PipeMorph / {method} = {r}")
    opt_ratio[method] = r
    x = np.arange(len(SEQLENS)) + (len(METHODS) - 1) / 2 * WIDTH
    plt.xticks(x, SEQLENS, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Time Per\nIteration (s)', fontsize=12)
    plt.grid(linestyle='-.')
plt.ylim(0, 2.6)
legend = plt.figlegend(loc=(0.1, 0.88), ncols=4, fontsize=12)
# plt.tight_layout(pad=0.8)
plt.savefig(f'{PATH}/seqlen.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.savefig(f'{PATH}/seqlen.png', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
for method in opt_ratio:
    print(method)
    print(f'avg {sum(opt_ratio[method]) / len(opt_ratio[method]):.3f}')
    print(f'max {max(opt_ratio[method]):.3f}\n')