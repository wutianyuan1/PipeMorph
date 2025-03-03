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
    all_log_data = []  # shape: (num_stages, num_iters, F+B per iter, 4)
    all_iter_times = []
    for i in range(num_stages):
        with open(f"{path}/GPU{i * dp_tp_prod}_rank{i * dp_tp_prod}_1F1B.log", 'r') as f:
            all_logs = f.read().split("Iteration")
            stage_all_log_data = []
            stage_iter_times = []
            for iter_log in all_logs:
                iter_log_data = []
                is_eval = True
                num_microbatches = 0
                x, exec_time, start_iter_time = float("inf"), 0, -1
                for log in iter_log.split("\n"):
                    terms = log.split()
                    if len(terms) < 2:
                        continue
                    if terms[1][-1] not in ['F', 'B']:
                        continue
                    if terms[1][-1] == 'F':
                        num_microbatches += 1
                    else:
                        is_eval = False
                    fbw2id = {'F':0, 'B':1}
                    # Single F/B
                    x, y = float(terms[3]), num_stages - 1 - i
                    if start_iter_time == -1:
                        start_iter_time = x
                    exec_time = float(terms[10]) # float(terms[6]) - float(terms[3])
                    iter_log_data.append([x, y, exec_time, fbw2id[terms[1][-1]]])
                stage_iter_times.append(x + exec_time - start_iter_time) # The last B is the iteration end
                if len(iter_log_data) != 0 and (not is_eval):
                    stage_all_log_data.append(iter_log_data)
        stage_all_log_data = np.array(stage_all_log_data)
        all_log_data.append(stage_all_log_data)

    all_log_data = np.array(all_log_data)[:, -count_last_iters :, :, :]
    for iter in range(all_log_data.shape[1]):
        iter_start_min = np.min(all_log_data[:, iter, 0, 0])
        all_log_data[:, iter, :, 0] -= iter_start_min
        all_iter_times.append(np.max(all_log_data[:, iter, -1, 0] + all_log_data[:, iter, -1, 2]))

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


if __name__ == '__main__':
    plt.figure(figsize=(10, 2.5))
    ax = plt.subplot(111)
    maxt, num_microbatches = parse_and_plot(ax, 2, 5, False)
    plt.savefig(f"{path}/real.png")