import os
import re
import numpy as np
import matplotlib.pyplot as plt

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/dpvspp'
MODEL = '7B'
METHODS = ['DP', 'ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
DELAYS = ['0.01', '0.02', '0.03']
LINESTYLES = ['-o', '-v', '-x', '-D']
HATCHES = ['//', '--', '\\\\', '||']
COUNT_LAST_ITERS = 5
WIDTH = 0.2

with open("experiments/overhead/times.txt", 'r') as f:
    zero_delay_times = eval(f.read())

plt.figure(figsize=(7, 2.5))
slowlink = 'Last'
method_times= []
x = np.arange(len(DELAYS))
for method in METHODS:
    times = []
    std_times = []
    for i, delay in enumerate(DELAYS):
        if delay != '0':
            dir = os.path.join(PATH, MODEL, method, '2_3' if method != 'DP' else '.', delay)
            try:
                assert os.path.exists(f"{dir}/real.png")
            except:
                print(f"{slowlink, delay, MODEL, method}")
            try:
                with open(f"{dir}/log.txt", 'r') as f:
                    iters = f.readlines()[-COUNT_LAST_ITERS :]
                    iters = [iter.split(' | ')[2] for iter in iters]
                    iter_times = np.array([float(re.findall(pattern, iter)[0]) for iter in iters]) / 1000.0
                    avg_iter_time = np.mean(iter_times)
                    std_iter_time = np.std(iter_times)
                    # print(slowlink, delay, model, method, avg_iter_time, std_iter_time)
                    times.append(avg_iter_time)
                    std_times.append(std_iter_time)
            except:
                    times.append(0)
            plt.text(x[i], times[-1] + 1.5*std_times[-1], "{:.2f}".format(times[-1]), fontdict={"fontsize": 12}, va='bottom', ha='center', zorder=100)
        else:
            times.append(zero_delay_times[method][1])
    if method == 'ZB-CPU':
        label = 'PipeMorph-CPU'
    elif method == 'ZB-CPU-ReSchedule':
        label = 'PipeMorph'
    elif method == 'ZB':
        label = 'GreyHound'
    elif method == 'DP':
        label = 'ZB'
    else:
        label = method
    plt.bar(x, times, width=WIDTH, label=label, zorder=100, edgecolor='black', hatch=HATCHES[METHODS.index(method)])
    plt.errorbar(x, times, yerr=std_times, color='black', fmt='o', ecolor='black', capsize=5, zorder=100)
    x = x + WIDTH
    method_times.append(times)
method_times = np.array(method_times)
plt.xticks(x - 0.5, DELAYS, fontsize=14)
plt.yticks(fontsize=14)
plt.yscale("log")
plt.xlabel('Network Delay (s)', fontsize=14)
plt.ylabel('Time Per\nIteration (s)', fontsize=14)
plt.grid(linestyle='-.')
plt.ylim(0.1, 600)
legend = plt.legend(fontsize=12, ncols=2, loc='upper left')
plt.tight_layout(pad=1.05)
plt.savefig(f'{PATH}/DP.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
