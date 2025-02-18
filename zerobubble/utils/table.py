import matplotlib.pyplot as plt
simu = {}
fused = {}
unfused = {}

with open("./table", 'r') as f:
    lines = f.readlines()[6:10]
    for line in lines:
        terms = line.split()
        for i, d in zip(range(1, 4), [simu, fused, unfused]):
            d[terms[0]] = int(terms[i])
for d, label in zip([unfused, fused, simu], ['Unfused', 'Fused', 'Simulation']):
    print(d)
    plt.bar(d.keys(), d.values(), label=label)
plt.legend(loc='lower right')
plt.savefig('./table.png')
