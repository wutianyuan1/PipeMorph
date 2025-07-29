import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ae.parse_time import parse_iter_times

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'ae/fig15'
MODEL = '7B'
METHODS = ['1F1B', 'ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
DELAYS = ['0', '0.02', '0.04', '0.06', '0.08', '0.10', '0.12']
LINESTYLES = ['-o', '-.v', '--x', '-.D']
COUNT_LAST_ITERS = 5
COLORS = ['#9673A6', '#E7AB10', '#6C8EBF', '#B85450']

with open("ae/fig20/times.txt", 'r') as f:
    zero_delay_times = eval(f.read())

plt.figure(figsize=(7, 2))
slowlink = 'Last'
method_times= []
for method in METHODS:
    x = np.arange(len(DELAYS))
    times = []
    for i, delay in enumerate(DELAYS):
        if delay != '0':
            dir = os.path.join(PATH, method, '0_1' if slowlink == 'First' else ('2_3' if MODEL in ['7B'] else '6_7'), delay)
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
            except:
                    times.append(0)
        else:
            times.append(zero_delay_times[method][0])
    if method == 'ZB-CPU':
        label = 'PipeMorph-CPU'
    elif method == 'ZB-CPU-ReSchedule':
        label = 'PipeMorph'
    else:
        label = method
    plt.plot(x, times, LINESTYLES[METHODS.index(method)], label=label, linewidth=3, color=COLORS[METHODS.index(method)])
    method_times.append(times)
method_times = np.array(method_times)
plt.xticks(x, DELAYS, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Communication Delay (s) between PP stages 6$\leftrightarrow$7', fontsize=14)
plt.ylabel('Time Per\nIteration (s)', fontsize=14)
plt.grid(linestyle='-.')
legend = plt.legend(fontsize=12)
plt.savefig(f'{PATH}/fig15.png', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.savefig(f'{PATH}/fig15.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
for i, method in enumerate(METHODS):
    print(f'1 - {method} / ZB = {1 - method_times[i][-1] / method_times[1][-1]:.3f}')
for i, method in enumerate(METHODS):
    print(f'{method} t_60ms / t_0ms - 1 = {method_times[i][-1]} / {method_times[i][0]} - 1 = {method_times[i][-1] / method_times[i][0] - 1:.3f}')