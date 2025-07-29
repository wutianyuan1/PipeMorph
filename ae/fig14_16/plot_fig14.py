import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_stages', type=int, default=4)
parser.add_argument('--dp_tp_prod', type=int, default=1)
args = parser.parse_args()

METHODS = ['ZB', 'ZB-CPU', 'ZB-CPU-ReSchedule']
XLIMIT = 2500

def parse_and_plot(idx, ax, path, warm_up_iters: int = 2, count_last_iters: int = 5, is_cpu: bool = False):
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
    comp_time = np.sum(all_log_data[:, :, 2])
    print(f"Bubble rate: {1 - comp_time / (all_log_data.shape[0] * all_iter_times[best_iter_id])}")
    maxt = 0

    labels = {0: 'Forward (F)', 1: 'Backward Input (B)', 2: 'Backward Weight (W)'}
    for stage in range(len(all_log_data)):
        for (x, y, exec_time, fbw_id) in all_log_data[stage]:
            if int(fbw_id) in labels and idx == 0:
                label = labels.pop(int(fbw_id))
            else:
                label = ''
            rect = patches.Rectangle((x, y), exec_time, 1, linewidth=1, edgecolor='black', facecolor=['#FCCCB3', '#CBE4E4', '#FBE7A3'][int(fbw_id)], label=label)
            ax.add_patch(rect)
            # ax.text(x + exec_time / 4, y + 1 / 2, fbw[int(fbw_id)])
            maxt = max(maxt, x + exec_time)
    rect = patches.Rectangle((0, 0), XLIMIT, num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
    ax.add_patch(rect)
    ax.set_title(f"{['ZB', 'PipeMorph-CPU', 'PipeMorph'][idx]}", fontsize=16)
    ax.set_xlim(0, XLIMIT)
    ax.set_ylim(0, num_stages)
    ax.set_yticks(np.arange(num_stages) + 0.5, [i for i in range(num_stages - 1, -1, -1)], fontsize=14)
    if idx == 1:
        plt.ylabel("Stages", fontsize=16)
    if idx == 2:
        plt.xlabel("Time (ms)", fontsize=16)
    if idx != 2:
        plt.xticks([])
    else:
        plt.xticks(fontsize=14)
    return maxt, num_microbatches

if __name__ == '__main__':

    plt.figure(figsize=(8, 4.5))
    for i, method in enumerate(METHODS):
        path = f"ae/fig14_16/{method}/2_3/0.06"
        ax = plt.subplot(int(f'31{i + 1}'))
        is_cpu = method != 'ZB'
        maxt, num_microbatches = parse_and_plot(i, ax, path, 2, 5, is_cpu)
    legend = plt.figlegend(loc=(0.04, 0.935), ncols=3, fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(f"ae/fig14_16/fig14.png", bbox_inches='tight', bbox_extra_artists=(legend,))
    plt.savefig(f"ae/fig14_16/fig14.pdf", bbox_inches='tight', bbox_extra_artists=(legend,))
