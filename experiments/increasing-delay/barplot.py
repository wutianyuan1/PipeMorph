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
COLORS2 = ['#462356', '#975B00', '#1C3E6F', '#780400']

with open("experiments/overhead/times.txt", 'r') as f:
    zero_delay_times = eval(f.read())

plt.figure(figsize=(7, 1.5))
slowlink = 'Last'
method_times= []
for method in METHODS:
    x = np.arange(len(DELAYS))
    times = []
    for i, delay in enumerate(DELAYS):
        old_method = method
        if method == 'ZB-CPU':
            method = 'ZB-CPU-old'
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
                    print(slowlink, delay, method, avg_iter_time, std_iter_time)
                    times.append(avg_iter_time)
            except:
                    times.append(0)
        else:
            times.append(zero_delay_times[old_method][1])
    if method == 'ZB-CPU':
        label = 'PipeMorph-CPU'
    elif method == 'ZB-CPU-ReSchedule':
        label = 'PipeMorph'
    else:
        label = method
    method_times.append(times[::2])
### TODO: 1F1B-CPU???
method_times.append([0.86208, 1.05292, 1.35618, 1.69054])
method_times = np.array(method_times)
num_data = len(method_times[0])
width = 0.3

# Base part
plt.bar(np.arange(num_data), [method_times[0, 0]] * num_data, width=width, bottom=0, zorder=100, label='1F1B-base', color=COLORS[0], alpha=1.0, edgecolor=COLORS2[0], hatch='//')
plt.bar(np.arange(num_data) + width, [method_times[1, 0]] * num_data, width=width, bottom=0, zorder=100, label='ZB-base', color=COLORS[0], alpha=0.7, edgecolor=COLORS2[0], hatch='\\\\')

# 1F1B/ZB-CPU - corresponding health, indicating bubble (dependency)
plt.bar(np.arange(num_data), method_times[4] - method_times[0, 0], width=width, bottom=method_times[0, 0], zorder=100, label='1F1B-bubble', color=COLORS[2], alpha=1.0, edgecolor=COLORS2[2], hatch='||')
plt.bar(np.arange(num_data) + width, method_times[2] - method_times[1, 0], width=width, bottom=method_times[1, 0], zorder=100, label='ZB-bubble', color=COLORS[2], alpha=0.7, edgecolor=COLORS2[2], hatch='--')


# 1F1B/ZB - correspondingCPU, indicating kernel blocking
plt.bar(np.arange(num_data), method_times[0] - method_times[4], width=width, bottom=method_times[2], zorder=100, label='1F1B-blocking', color=COLORS[1], alpha=1.0, edgecolor=COLORS2[1], hatch='**')
plt.bar(np.arange(num_data) + width, method_times[1] - method_times[2], width=width, bottom=method_times[2], zorder=100, label='ZB-blocking', color=COLORS[1], alpha=0.7, edgecolor=COLORS2[1], hatch='..')

for i in range(num_data):
    plt.text(i - width/2, method_times[0, i] + 0.05, "{:.2f}".format(method_times[0, i]), fontdict={"fontsize": 14})
    plt.text(i + width/2, method_times[1, i] + 0.05, "{:.2f}".format(method_times[1, i]), fontdict={"fontsize": 14})
plt.ylim(0, 3.5)

print(method_times)
plt.xticks(x[::2]/2 + width/2, DELAYS[::2], fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 3.8)
plt.xlabel('Communication Delay (s) between PP stages 6$\leftrightarrow$7', fontsize=14)
plt.ylabel('Time Per\nIteration (s)', fontsize=14)
plt.grid(linestyle='-.')
legend = plt.figlegend(loc=(0.15, 0.78), ncols=3, fontsize=12, frameon=False)
plt.savefig(f'{PATH}/{MODEL}bar.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
for i, method in enumerate(METHODS):
    print(f'1 - {method} / ZB = {1 - method_times[i][-1] / method_times[1][-1]:.3f}')
for i, method in enumerate(METHODS):
    print(f'{method} t_60ms / t_0ms - 1 = {method_times[i][-1]} / {method_times[i][0]} - 1 = {method_times[i][-1] / method_times[i][0] - 1:.3f}')