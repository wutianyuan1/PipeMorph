import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_stages', type=int)
args = parser.parse_args()

for dp, gpus in enumerate([[0, 2], [1, 3]]):
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(111)
    maxt = 0
    num_stages = args.num_stages
    for gpu, i in zip(gpus, range(num_stages)):
        with open(f"GPU{gpu}_rank{i}.log", 'r') as f:
            num_microbatches = 0
            for log in f.readlines():
                terms = log.split()
                if terms[1][-1] not in ['F', 'B', 'W']:
                    continue
                if terms[1][-1] == 'F':
                    num_microbatches += 1
                x, y = int(terms[3]), num_stages - 1 - i
                exec_time = int(terms[6]) - int(terms[3])
                rect = patches.Rectangle((x, y), exec_time, 1, linewidth=1, edgecolor='black', facecolor={'F': '#FCCCB3', 'W': '#FBE7A3', 'B': '#CBE4E4'}[terms[1][-1]])
                ax.add_patch(rect)
                ax.text(x + exec_time / 4, y + 1 / 2, terms[1])
                maxt = max(maxt, int(terms[6]))
    rect = patches.Rectangle((0, 0), maxt, num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
    ax.add_patch(rect)
    ax.set_title(f"S={num_stages}, B={num_microbatches}, Total Time = {maxt}")
    ax.set_xlim(0, maxt)
    ax.set_ylim(0, num_stages)
    ax.set_yticks(np.arange(num_stages) + 0.5, [f"Stage {i}" for i in range(num_stages - 1, -1, -1)])
    plt.savefig(f"/workspace/PipeMorph/zerobubble/real_dp{dp}.png")
