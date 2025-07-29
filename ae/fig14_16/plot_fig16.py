import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ae.parse_time import parse_iter_times

pattern = r"[-+]?\d*\.\d+|\d+"

PATH = 'ae/fig14_16'
MODELS = ['7B']
METHODS = ['1F1B', 'ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
DELAYS = ['0.06', '0.12']
HATCHES = ['//', '--', '\\\\', '||']
COUNT_LAST_ITERS = 5
WIDTH = 0.23
# COLORS = ['#E1D5E7', '#F8CECC', '#FFE6CC', '#DAE8FC']
COLORS = ['#9673A6', '#E7AB10', '#6C8EBF', '#B85450']
COLORS2 = ['#462356', '#975B00', '#1C3E6F', '#780400']

opt_ratio = {method: {slowlink: {delay: None for delay in DELAYS} for slowlink in ['First', 'Last']} for method in METHODS}
t = {method: {slowlink: {delay: None for delay in DELAYS} for slowlink in ['First', 'Last']} for method in METHODS}

plt.figure(figsize=(14, 3.6))
for sid, slowlink in enumerate(['First', 'Last']):
    for figid, delay in enumerate(DELAYS):
        x = np.arange(len(MODELS))
        plt.subplot(2, 2, 2*sid + figid + 1)
        method_times = []
        for method in METHODS:
            times = []
            std_times = []
            for i, model in enumerate(MODELS):
                dir = os.path.join(PATH, method, '0_1' if slowlink == 'First' else ('2_3' if model in ['7B'] else '6_7'), delay)
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
            plt.bar(x, times, width=WIDTH, label=label if sid == 0 and figid == 0 else None, zorder=100, alpha=0.7, edgecolor=COLORS2[METHODS.index(method)], hatch=HATCHES[METHODS.index(method)], color=COLORS[METHODS.index(method)])
            plt.errorbar(x, times, yerr=std_times, color='black', fmt='o', ecolor='black', capsize=5, zorder=100)
            x = x + WIDTH
            method_times.append(times)
        method_times = np.array(method_times)
        # print(f"{slowlink}, {delay}: Optimize ratio: {method_times[1] / method_times[3]}")  # ZB / ours
        for i, method in enumerate(METHODS):
            r = 1 - method_times[3] / method_times[i]
            print(f"{slowlink}, {delay}:  1 - PipeMorph / {method} = {r}")
            opt_ratio[method][slowlink][delay] = r
            t[method][slowlink][delay] = method_times[i]
        x = np.arange(len(MODELS)) + (len(METHODS) - 1) / 2 * WIDTH
        plt.xticks(x, MODELS, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(f"{'120 ms' if delay == '0.12' else '60 ms'} delay between {slowlink.lower()} two stages", fontsize=14)
        if figid == 0:
            plt.ylabel('Time Per\nIteration (s)', fontsize=12)
        # plt.title(f'{float(delay) * 1000}ms Delay on {slowlink} Link')
        plt.grid(linestyle='-.')
plt.tight_layout(pad=1.05)
legend = plt.figlegend(loc=(0.27, 0.9), ncols=4, fontsize=14, frameon=False)
plt.savefig(f'{PATH}/fig16.pdf', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.savefig(f'{PATH}/fig16.png', bbox_inches='tight', bbox_extra_artists=(legend,))
plt.close()
# for method in opt_ratio:
#     print(method)
#     for i, model in enumerate(MODELS):
#         vals = [opt_ratio[method][slowlink][delay][i] for slowlink in ['First', 'Last'] for delay in DELAYS]
#         print(model, vals)
#         print(f'avg {sum(vals) / len(vals):.3f}')
#         print(f'max {max(vals):.3f}')
#         print()

# link
for link in ['First', 'Last']:
    ppm_cpu = opt_ratio['ZB-CPU'][link]['0.12']
    print('1 - PipeMorph / PipeMorph-CPU', ppm_cpu)
    print('1 - PipeMorph / PipeMorph-CPU', 1 - t['ZB-CPU-ReSchedule'][link]['0.12'] / t['ZB-CPU'][link]['0.12'])
    print(f'avg {sum(ppm_cpu) / len(ppm_cpu):.3f}')
    print(f'max {max(ppm_cpu):.3f}')
    print()
# delay
for method in METHODS:
    vals1 = np.concatenate([t[method][slowlink]['0.06'] for slowlink in ['First', 'Last']], axis=0)
    vals2 = np.concatenate([t[method][slowlink]['0.12'] for slowlink in ['First', 'Last']], axis=0)
    print(vals1)
    print(vals2)
    print(f'{method} avg {np.mean(vals2 / vals1):.3f}')
    print()