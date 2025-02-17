import os

nodelist = os.environ['SLURM_NODELIST']
assert nodelist[:5] == 'dgx-['
nodelist = nodelist[5:-1].split(",")
nodes = []
for item in nodelist:
    if '-' not in item:
        nodes.append(int(item))
    else:
        start, end = item.split("-")
        nodes += list(range(int(start), int(end) + 1))
print(f"dgx-{nodes[0]}")