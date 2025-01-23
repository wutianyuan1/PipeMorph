import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_stages', type=int, default=4)
args = parser.parse_args()

logs_list = []
plt.figure(figsize=(10, 6))
ax = plt.subplot(211)
maxt = 0
num_stages = args.num_stages
for i in range(num_stages):
    with open(f"GPU{i}_rank{i}.log", 'r') as f:
        num_microbatches = 0
        for log in f.readlines():
            terms = log.split()
            if terms[1][-1] not in ['F', 'B', 'W']:
                continue
            if terms[1][-1] == 'F':
                num_microbatches += 1
            x, y = float(terms[3]), num_stages - 1 - i
            exec_time = float(terms[6]) - float(terms[3])
            rect = patches.Rectangle((x, y), exec_time, 1, linewidth=1, edgecolor='black', facecolor={'F': '#FCCCB3', 'W': '#FBE7A3', 'B': '#CBE4E4'}[terms[1][-1]])
            ax.add_patch(rect)
            ax.text(x + exec_time / 4, y + 1 / 2, terms[1])
            maxt = max(maxt, float(terms[3]) + exec_time)
rect = patches.Rectangle((0, 0), maxt, num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
ax.add_patch(rect)
ax.set_title(f"S={num_stages}, B={num_microbatches}, Total Time = {maxt}")
ax.set_xlim(0, maxt)
ax.set_ylim(0, num_stages)
ax.set_yticks(np.arange(num_stages) + 0.5, [f"Stage {i}" for i in range(num_stages - 1, -1, -1)])

# Simulation plot
ax = plt.subplot(212)
maxt2 = 0
with open("simu.txt", 'r') as f:
    for log in f.readlines():
        terms = log.strip("\n").split(', ')
        if terms[1] not in ['F', 'B', 'W']:
            continue
        x, y = float(terms[2]), num_stages - 1 - float(terms[0])
        exec_time = float(terms[3]) - float(terms[2])
        rect = patches.Rectangle((x, y), exec_time, 1, linewidth=1, edgecolor='black', facecolor={'F': 'red', 'B': 'green', 'W': 'blue'}[terms[1]], zorder=101, alpha=0.2)
        ax.add_patch(rect)
        ax.text(x + exec_time / 4 if terms[1] != 'W' else x + exec_time / 10, y + 1 / 2, terms[1])
        maxt2 = max(maxt2, float(terms[3]))
rect = patches.Rectangle((0, 0), max(maxt, maxt2), num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
ax.add_patch(rect)
ax.set_title(f"S={num_stages}, B={num_microbatches}, Total Time = {maxt2}")
ax.set_xlim(0, maxt)
ax.set_ylim(0, num_stages)
ax.set_yticks(np.arange(num_stages) + 0.5, [f"Stage {i}" for i in range(num_stages - 1, -1, -1)])
plt.savefig("/workspace/test-varuna/zerobubble/real.png")
