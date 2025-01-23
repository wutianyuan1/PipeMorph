import matplotlib.pyplot as plt
import numpy as np

colors = ["#FBE7A3", "#CBE4E4", "#8FAADC", "#FCCCB3", "#F2F2F2"]
offset = 0
barwidth = 0.3
results = {"zb": {}, "1f1b": {}}
maxy = 0
for policy, result in results.items():
    with open(f"results/{policy}/result_{policy}.txt", "r") as file:
        for line in file.readlines()[1:]:
            terms = line.split()
            assert terms[0] not in result
            result[terms[0]] = int(terms[-1])
    baseline_color = colors[4] if offset == 0 else colors[1]
    # [colors[1]] * 3 + [colors[2]] * 3 + [colors[3]]
    slow_color = [colors[3]] * (len(result) - 1) if offset == 0 else [colors[0]] * (len(result) - 1)
    slow_label = 'ZB-Slow' if offset == 0 else '1F1B-Slow'
    baseline_label = 'ZB-Baseline' if offset == 0 else '1F1B-Baseline'
    ys = list(result.values())
    plt.bar(np.arange(len(result) - 1) + offset, ys[1:], color=slow_color, edgecolor='#101010', width=barwidth, label=slow_label)
    plt.bar(np.arange(len(result) - 1) + offset, [ys[0]] * (len(result) - 1), color=baseline_color, edgecolor='#101010', width=barwidth, label=baseline_label)
    for i in range(1, len(result)):
        plt.text(i + offset - 1.3, ys[i] + 100, f"+{ys[i] - ys[0]}", fontsize=12)
    offset += barwidth
    maxy = max(maxy, max(result.values()) * 1.4)


plt.legend(ncols=2)
plt.ylim(0, maxy)
tmp = list(result.keys())[1:]
xticks = []
for i in tmp:
    m = i.split('-')
    m = [j[0] + '->' + j[1] for j in m]
    xticks.append('\n'.join(m))

plt.xticks(np.arange(len(result) - 1) + barwidth / 2, xticks, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Slow Links", fontsize=14)
plt.ylabel("Time per Iteration (ms)", fontsize=14)
plt.tight_layout()
plt.savefig(f"results/cmp2.png")
    
for (policy, result), color in zip(reversed(results.items()), [colors[0], colors[2]]):
    plt.bar(result.keys(), result.values(), color=color, label=policy)
plt.ylim(0, max(max(results["1f1b"].values()), max(results["zb"].values())) * 1.1)
plt.xlabel("Slow Links")
plt.ylabel("Time per Iteration (ms)")
plt.legend()
plt.savefig("results/cmp.png")
plt.close()