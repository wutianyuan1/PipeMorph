import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


logs_list = []
plt.figure(figsize=(10, 6))
ax = plt.subplot(211)
maxt = 0
num_stages = 2
for i in range(num_stages):
    with open(f"log_{i}.log", 'r') as f:
        num_microbatches = 0
        for log in f.readlines():
            terms = log.split()
            if terms[1][-1] not in ['F', 'B', 'W']:
                continue
            if terms[1][-1] == 'F':
                num_microbatches += 1
            x, y = float(terms[3]), num_stages - 1 - i
            exec_time = float(terms[6]) - float(terms[3])
            print(terms[3], exec_time)
            rect = patches.Rectangle((x, y), exec_time, 1, linewidth=1, edgecolor='black', facecolor={'F': '#FCCCB3', 'W': '#FBE7A3', 'B': '#CBE4E4'}[terms[1][-1]])
            ax.add_patch(rect)
            ax.text(x + exec_time / 4, y + 1 / 2, terms[1])
            maxt = max(maxt, float(terms[6]))
rect = patches.Rectangle((0, 0), maxt, num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
ax.add_patch(rect)
ax.set_title(f"S={num_stages}, B={num_microbatches}, Total Time = {maxt}")
ax.set_xlim(0, maxt)
ax.set_ylim(0, num_stages)
ax.set_yticks(np.arange(num_stages) + 0.5, [f"Stage {i}" for i in range(num_stages - 1, -1, -1)])
plt.savefig(f"/workspace/PipeMorph/test/test.png")