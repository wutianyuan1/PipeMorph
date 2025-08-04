import numpy as np
import re


def parse_iter_times_log(path: str, method: str, count_iters_range: tuple):
    with open(f"{path}/log.txt", 'r') as f:
        content = f.read()
    all_iter_times = [float(i) for i in re.findall(r'per iteration \(ms\): (\d+\.\d+)', content)]
    return np.array(all_iter_times[count_iters_range[0]:count_iters_range[1]])


def parse_iter_times(path: str, method: str, count_iters_range: tuple):
    suffix_map = {'1F1B': '_1F1B', 'ZB': '', 'ZB-CPU': '_CPU', 'ZB-CPU-ReSchedule': '_CPU', 'ALL-CPU': '_CPU'}
    suffix = suffix_map[method]
    num_stages = 4
    dp_tp_prod = 1
    all_iter_times = []
    for i in range(num_stages):
        with open(f"{path}/GPU{i * dp_tp_prod}_rank{i * dp_tp_prod}{suffix}.log", 'r') as f:
            all_logs = f.read().split("Iteration")
            stage_iter_times = []
            for iter_log in all_logs:
                is_eval = True
                log_per_ops = iter_log.split("\n")
                start, num_bwd = None, 0 # For 1F1B
                for j, log in enumerate(log_per_ops):
                    terms = log.split()
                    if len(terms) < 2:
                        continue
                    if terms[1][-1] not in ['F', 'B', 'W']:
                        continue
                    if terms[1][-1] != 'F':
                        is_eval = False
                    # For 1F1B
                    if start is None:
                        start = float(terms[3])
                    if terms[1][-1] == 'B':
                        num_bwd += 1
                    if method != '1F1B': # There is no 'W' in logs of 1F1B as it does not split input/weight backward
                        # Multiple Ws e.g., 4W
                        if len(terms[1]) != 1:
                            assert not is_eval
                            stage_iter_times.append(float(terms[3]) + float(terms[10])) # The last end in nW is the iteration end
                    else:
                        if j == len(log_per_ops) - 2 and not is_eval:
                            assert terms[1][-1] == 'B' and num_bwd == 12
                            stage_iter_times.append(float(terms[3]) + float(terms[10]) - start)
        all_iter_times.append(stage_iter_times)

    all_iter_times = np.array(all_iter_times)[:, count_iters_range[0] : count_iters_range[1]]
    all_iter_times = np.max(all_iter_times, axis=0)  # max across all stages to get the real iteration time

    best_iter_id = np.argmin(all_iter_times)
    mean_iter_time = np.mean(all_iter_times)
    print(f"Min time iter is {best_iter_id}, min iter time = {all_iter_times[best_iter_id]}, mean iter time = {mean_iter_time}")
    return all_iter_times
