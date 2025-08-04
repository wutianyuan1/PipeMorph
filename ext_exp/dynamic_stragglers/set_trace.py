import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from typing import List
from typing import List, Tuple, Dict

np.random.seed(2025)

parser = argparse.ArgumentParser()
parser.add_argument("--num_stages", type=int, default=4)
parser.add_argument("--min_duration", type=int, default=100)
parser.add_argument("--max_duration", type=int, default=300)
args = parser.parse_args()

LINKS = [f'{i}_{i + 1}' for i in range(args.num_stages - 1)]
# Distribution parameters
MIN_DURATION, MAX_DURATION = args.min_duration, args.max_duration # Duration ~ U(MIN_DURATION, MAX_DURATION)
DELAY_MEAN, DELAY_STD = 30, 10 # Delay ~ N(DELAY_MEAN, DELAY_STD)
# We start from the 8-th iteration, the training lasts for 15 iterations,
# and we assume the maximum iteration time across 4 methods is 1500 ms.
# So we maximize the number of changes to fully cover the 8th to 15th iterations.
NUM_CHANGE_TIMINGS = 1500 * (15 - 7) // MIN_DURATION

trace: List[Tuple[float, Dict[str, float]]] = [] # (timing, link2delay)

timing = 0
link2delay = {link: 0.0 for link in LINKS} # link: delay

for i in range(NUM_CHANGE_TIMINGS):
    # Sample links to change delays
    num_links_to_change = np.random.randint(1, args.num_stages)
    links_to_change = [LINKS[i] for i in np.random.choice(args.num_stages - 1, num_links_to_change, replace=False)]
    # Sample delays
    delays = np.random.randn(num_links_to_change) * DELAY_STD + DELAY_MEAN
    delays = [float(l) if l > 0 else 0.0 for l in delays]
    # Sample timing
    timing += np.random.rand() * (MAX_DURATION - MIN_DURATION) + MIN_DURATION

    for link, delay in zip(links_to_change, delays):
        link2delay[link] = delay
    trace.append((timing, copy.deepcopy(link2delay)))
    print(f"[{i}] Timing: {timing}\nChanged links: {links_to_change}\nLink to Delay: {link2delay}")

for i in range(args.num_stages - 1):
    data = [(timing, link2delay[LINKS[i]]) for timing, link2delay in trace]
    plt.subplot(args.num_stages - 1, 1, i + 1)
    plt.plot([0] + [x / 1000 for x, _ in data], [0] + [y for _, y in data], label=f'Link {LINKS[i]}')
    # y_min, y_max = plt.ylim()
    # plt.vlines([0] + [x / 1000 for x, _ in data], [y_min] * (len(data) + 1), [y_max] * (len(data) + 1), linestyles='--')
    plt.legend()
    if i == args.num_stages - 2:
        plt.xlabel('Wallclock Time (s)')
    if i == (args.num_stages - 1) // 2:
        plt.ylabel('Delay (ms)')
# plt.legend()
# plt.xlabel('Wallclock Time (s)')
# plt.ylabel('Delay (ms)')
USER_NAME = os.getenv("SLURM_JOB_USER")
plt.savefig(F"/home/{USER_NAME}/workspace/PipeMorph-Ext-Exp/ext_exp/dynamic_stragglers/trace_{MIN_DURATION}-{MAX_DURATION}.png")
trace_lines = [f"{timing / 1000};{','.join(link2delay.keys())};{','.join([str(t / 1000) for t in link2delay.values()])}\n" for timing, link2delay in trace]
with open(f"/home/{USER_NAME}/workspace/PipeMorph-Ext-Exp/ext_exp/dynamic_stragglers/{MIN_DURATION}-{MAX_DURATION}.trace", 'w') as f:
    f.writelines(trace_lines)
with open(f"/home/{USER_NAME}/workspace/PipeMorph-Ext-Exp/zerobubble/megatron/core/failslow_injection/slowlinks.trace", 'w') as f:
    f.writelines(trace_lines)