import os
import re
import numpy as np
import matplotlib.pyplot as plt

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'experiments/increasing-delay'
MODEL = '14B'
METHODS = ['1F1B', 'ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
DELAYS = ['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06']
LINESTYLES = ['-o', '-.v', '--x', '-.D']
COUNT_LAST_ITERS = 5
COLORS = ['#9673A6', '#E7AB10', '#6C8EBF', '#B85450']

with open("experiments/overhead/times.txt", 'r') as f:
    zero_delay_times = eval(f.read())

plt.figure(figsize=(7, 2))
slowlink = 'Last'
method_times= []
for method in METHODS:
    x = np.arange(len(DELAYS))
    times = []
    for i, delay in enumerate(DELAYS):
        if delay != '0':
            dir = os.path.join(PATH, MODEL, method, '0_1' if slowlink == 'First' else ('2_3' if MODEL in ['7B'] else '6_7'), delay)
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
            except:
                    times.append(0)
        else:
            times.append(zero_delay_times[method][1])
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
plt.savefig(f'{PATH}/{MODEL}.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
for i, method in enumerate(METHODS):
    print(f'1 - {method} / ZB = {1 - method_times[i][-1] / method_times[1][-1]:.3f}')
for i, method in enumerate(METHODS):
    print(f'{method} t_60ms / t_0ms - 1 = {method_times[i][-1]} / {method_times[i][0]} - 1 = {method_times[i][-1] / method_times[i][0] - 1:.3f}')