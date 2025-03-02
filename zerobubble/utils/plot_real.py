import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_stages', type=int, default=4)
parser.add_argument('--dp_tp_prod', type=int, default=1)
args = parser.parse_args()

path = os.getenv("OUT_DIR")
path = path if path is not None else '.'

def parse_and_plot(ax, warm_up_iters: int = 2, count_last_iters: int = 5, is_cpu: bool = False):
    num_stages = args.num_stages
    dp_tp_prod = args.dp_tp_prod
    all_log_data = []
    all_iter_times = []
    for i in range(num_stages):
        with open(f"{path}/GPU{i * dp_tp_prod}_rank{i * dp_tp_prod}{'_CPU' if is_cpu else ''}.log", 'r') as f:
            all_logs = f.read().split("Iteration")
            stage_all_log_data = []
            stage_iter_times = []
            for iter_log in all_logs:
                iter_log_data = []
                is_eval = True
                num_microbatches = 0
                for log in iter_log.split("\n"):
                    terms = log.split()
                    if len(terms) < 2:
                        continue
                    if terms[1][-1] not in ['F', 'B', 'W']:
                        continue
                    if terms[1][-1] == 'F':
                        num_microbatches += 1
                    else:
                        is_eval = False
                    fbw2id = {'F':0, 'B':1, 'W':2}
                    # Single F/B/W
                    if len(terms[1]) == 1:
                        x, y = float(terms[3]), num_stages - 1 - i
                        exec_time = float(terms[10]) # float(terms[6]) - float(terms[3])
                        iter_log_data.append([x, y, exec_time, fbw2id[terms[1][-1]]])
                    else:  # Multiple Ws e.g., 4W
                        x, y = float(terms[3]), num_stages - 1 - i
                        exec_time = float(terms[10]) # float(terms[6]) - float(terms[3])
                        nw = int(terms[1][:-1])
                        avg_t = exec_time / nw
                        for wid in range(nw):
                            iter_log_data.append([x, y, avg_t, fbw2id[terms[1][-1]]])
                            x += avg_t
                        stage_iter_times.append(x) # The last end in nW is the iteration end
                if len(iter_log_data) != 0 and (not is_eval):
                    stage_all_log_data.append(iter_log_data)
        stage_all_log_data = np.array(stage_all_log_data)
        all_log_data.append(stage_all_log_data)
        all_iter_times.append(stage_iter_times)

    all_log_data = np.array(all_log_data)[:, -count_last_iters :, :, :]
    all_iter_times = np.array(all_iter_times)[:, -count_last_iters :]
    all_iter_times = np.max(all_iter_times, axis=0)  # max across all stages to get the real iteration time
    # TODO: Madoka: current implementation is to find a best iteration to plot (i.e., min iteration time), consider using mean?
    best_iter_id = np.argmin(all_iter_times)
    print(f"Min time iter is {best_iter_id}, min iter time = {all_iter_times[best_iter_id]}")
    all_log_data = all_log_data[:, best_iter_id, :, :]
    maxt = 0
    for stage in range(len(all_log_data)):
        for (x, y, exec_time, fbw_id) in all_log_data[stage]:
            fbw = ['F', 'B', 'W']
            rect = patches.Rectangle((x, y), exec_time, 1, linewidth=1, edgecolor='black', facecolor=['#FCCCB3', '#CBE4E4', '#FBE7A3'][int(fbw_id)])
            ax.add_patch(rect)
            ax.text(x + exec_time / 4, y + 1 / 2, fbw[int(fbw_id)])
            maxt = max(maxt, x + exec_time)
    rect = patches.Rectangle((0, 0), maxt, num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
    ax.add_patch(rect)
    ax.set_title(f"S={num_stages}, B={num_microbatches}, Total Time = {maxt}")
    ax.set_xlim(0, maxt)
    ax.set_ylim(0, num_stages)
    ax.set_yticks(np.arange(num_stages) + 0.5, [f"Stage {i}" for i in range(num_stages - 1, -1, -1)])
    return maxt, num_microbatches

# Simulation plot
def plot_simulation(ax, maxt, num_microbatches):
    num_stages = args.num_stages
    maxt2 = 0
    with open(f"{path}/simu.txt", 'r') as f:
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
    ax.set_xlim(0, maxt2)
    ax.set_ylim(0, num_stages)
    ax.set_yticks(np.arange(num_stages) + 0.5, [f"Stage {i}" for i in range(num_stages - 1, -1, -1)])

if __name__ == '__main__':
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(311)
    maxt, num_microbatches = parse_and_plot(ax, 2, 5, False)
    ax = plt.subplot(312)
    maxt, num_microbatches = parse_and_plot(ax, 2, 5, True)
    ax = plt.subplot(313)
    plot_simulation(ax, maxt, num_microbatches)
    plt.tight_layout()
    plt.savefig(f"{path}/real.png")
