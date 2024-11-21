import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
from pipeline_simulator.batches import Batch, ForwardBatch, BackwardBatch, BackwardInputBatch, BackwardWeightBatch, BubbleBatch
from pipeline_simulator.batches import FORWARD_TIMES, BACKWARD_TIMES, SLOW_FACTORS
from pipeline_simulator.policy import PipelinePolicy, GpipePolicy, PipeDreamPolicy, LearnedPolicy, ZeroBubblePolicy, FixedPolicy


class PipelineSimulator(object):
    def __init__(self, num_stages: int, num_batches: int, policy: PipelinePolicy,
                 slow_stages: List[int], comm_delay: Dict[Tuple[int, int], int],
                 split_backward: bool = False) -> None:
        '''comm_delay: (i, j) -> T. delays T seconds between stage i and j'''
        self.num_stages = num_stages
        self.num_batches = num_batches
        self.split_backward = split_backward
        self.slow_stages = slow_stages
        self.comm_delay = comm_delay
        self.task_queues = [[] for _ in range(self.num_stages)]
        self.history_queues = [[] for _ in range(self.num_stages)]
        self.next_avail_times = np.zeros(self.num_stages, dtype=np.float32)
        self.policy = policy
        for i in range(self.num_batches):
            fail_slow = True if (0 in self.slow_stages) else False
            self.task_queues[0].append(ForwardBatch(i, fail_slow, -1, 0))

    def add_dependency(self, time: int, i: int, batch: Batch) -> None:
        if isinstance(batch, ForwardBatch):
            if i != self.num_stages - 1:
                fail_slow = True if i + 1 in self.slow_stages else False
                delay = self.comm_delay.get((i, i + 1), 0) + 1
                self.task_queues[i + 1].append(ForwardBatch(batch.batch_idx, fail_slow, time + delay, i + 1))
            else:
                fail_slow = True if i in self.slow_stages else False
                if self.split_backward:
                    self.task_queues[i].append(BackwardInputBatch(batch.batch_idx, fail_slow, time + 1, i))
                    self.task_queues[i].append(BackwardWeightBatch(batch.batch_idx, fail_slow, time + 1, i))
                else:
                    self.task_queues[i].append(BackwardBatch(batch.batch_idx, fail_slow, time + 1, i))
        elif isinstance(batch, BackwardBatch):
            if i != 0:
                fail_slow = True if i - 1 in self.slow_stages else False
                delay = self.comm_delay.get((i - 1, i), 0) + 1
                self.task_queues[i - 1].append(BackwardBatch(batch.batch_idx, fail_slow, time + delay, i - 1))
        elif isinstance(batch, BackwardInputBatch):
            fail_slow = True if i - 1 in self.slow_stages else False
            delay = self.comm_delay.get((i - 1, i), 0) + 1
            if i != 0:
                self.task_queues[i - 1].append(BackwardInputBatch(batch.batch_idx, fail_slow, time + delay, i - 1))
                self.task_queues[i - 1].append(BackwardWeightBatch(batch.batch_idx, fail_slow, time + delay, i - 1))
        elif isinstance(batch, BackwardWeightBatch) or isinstance(batch, BubbleBatch):
            pass
        else:
            raise RuntimeError(f"Unrecognized batch {batch}!")

    def simulate(self) -> int:
        time = 0
        while True:
            num_empty_queues = 0
            for i in range(self.num_stages):
                # No batches in queue
                if len(self.task_queues[i]) == 0:
                    # No previous batches => initial case
                    if len(self.history_queues[i]) == 0:
                        num_empty_queues += 1
                        continue
                    # If there are previously executed batches
                    else:
                        # If all previous batches are alreadly finished
                        if time > self.history_queues[i][-1].execution_time + self.history_queues[i][-1].execution_begin:
                            num_empty_queues += 1
                            continue

                # If this stage is busy now: case1 - before its last second, just do nothing and continue
                if time < self.next_avail_times[i] - 1:
                    continue
                # If this stage is busy now: case2 - at its last second, add its dependencies to corresponding task queues, and mark them can be executed >=t+1
                if time == self.next_avail_times[i] - 1:
                    self.add_dependency(time, i, self.history_queues[i][-1])
                    continue

                # If this stage is available (t >= self.next_avail_times[i])
                # If there are nothing in task queue (i.e., the last batch is executing), continue
                if len(self.task_queues[i]) == 0:
                    continue
                # If there are available tasks
                # print(f"===Stage={i}, task={self.task_queues[i]}, finish={self.history_queues[i]}")
                batch_pos = self.policy.pick_batch_to_run(self.task_queues[i], self.history_queues[i], time, i)
                if batch_pos is None or time < self.task_queues[i][batch_pos].min_begin_time:
                    continue
                batch = self.task_queues[i].pop(batch_pos)
                batch.execution_begin = time
                self.next_avail_times[i] = batch.execution_begin + batch.execution_time
                self.history_queues[i].append(batch)
                # print(f">>t={time}, stage={i}, start executing {batch}!")
            if num_empty_queues == self.num_stages:
                break
            time += 1
        return time

    def gen_schedule_no_comm_agg(self) -> List[List[str]]:
        schedules = []
        for i in range(self.num_stages):
            stage_schedule = []
            for j in range(len(self.history_queues[i])):
                batch = self.history_queues[i][j]
                batch_schedule = []
                if isinstance(batch, ForwardBatch):
                    if i != 0:
                        batch_schedule.append('RECV_FORWARD')
                    batch_schedule.append(batch.type)
                    if i != self.num_stages - 1:
                        batch_schedule.append('SEND_FORWARD')
                elif isinstance(batch, BackwardInputBatch) or isinstance(batch, BackwardBatch):
                    if i != self.num_stages - 1:
                        batch_schedule.append('RECV_BACKWARD')
                    batch_schedule.append(batch.type)
                    if i != 0:
                        batch_schedule.append('SEND_BACKWARD')
                elif isinstance(batch, BackwardWeightBatch):
                    batch_schedule.append(batch.type)
                stage_schedule += batch_schedule
            schedules.append(stage_schedule)
        return schedules

    def gen_schedule_graph_no_comm(self) -> List[int]:
        graph_complete_ts = np.zeros(self.num_batches * self.num_stages * 3, dtype=int)
        def get_id(batch: Batch, stage: int, bid: int) -> int:
            type_mapping = {ForwardBatch: 0, BackwardInputBatch: 1, BackwardWeightBatch: 2}
            return type_mapping[type(batch)] * (self.num_batches * self.num_stages) + stage * self.num_batches + bid

        for i in range(self.num_stages):
            for j in range(len(self.history_queues[i])):
                batch = self.history_queues[i][j]
                end_time = batch.execution_begin + batch.execution_time
                graph_complete_ts[get_id(batch, i, batch.batch_idx)] = end_time
        return graph_complete_ts.tolist()

    def plot(self) -> None:
        plt.figure(figsize=(10, 3))
        ax = plt.subplot(111)
        maxt = 0
        for i in range(self.num_stages):
            for j in range(len(self.history_queues[i])):
                batch = self.history_queues[i][j]
                batch.plot(ax, self.num_stages - i - 1, 1)
                maxt = max(maxt, batch.execution_begin + batch.execution_time)
        normal_stage_total = sum([(BACKWARD_TIMES[i] + FORWARD_TIMES[i]) * self.num_batches for i in range(self.num_stages) if i not in self.slow_stages])
        slow_stage_total = sum([(BACKWARD_TIMES[i] + FORWARD_TIMES[i]) * self.num_batches * SLOW_FACTORS[i] for i in range(self.num_stages) if i in self.slow_stages])
        usage = (normal_stage_total + slow_stage_total) / (maxt * self.num_stages)
        rect = patches.Rectangle((0, 0), maxt, self.num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
        ax.add_patch(rect)
        ax.set_title(f"S={self.num_stages}, B={self.num_batches}, Total Time = {maxt}, Bubble Rate = {(1 - usage) * 100:.2f}%")
        ax.set_xlim(0, maxt)
        ax.set_ylim(0, self.num_stages)
        ax.set_yticks(np.arange(self.num_stages) + 0.5, [f"Stage {i}" for i in range(self.num_stages - 1, -1, -1)])

    def export(self, fn: str) -> None:
        f = open(fn, 'w')
        for i in range(self.num_stages):
            f.write(" ".join([repr(j) for j in self.history_queues[i]]) + '\n')
        f.close()


def main() -> None:
    num_stages, num_batches = 4, 12
    # Please comment out the unused policies
    # policy = PipeDreamPolicy(num_stages)
    # policy = LearnedPolicy(num_stages, num_batches)
    # policy = GpipePolicy()
    # policy = ZeroBubblePolicy(num_stages)
    policy = FixedPolicy(num_stages, "./pipeline_simulator/originalzb.txt")
    comm_delay = {
        (0, 1): 100,
    }
    simulator = PipelineSimulator(num_stages, num_batches, policy, [], comm_delay, True)
    simulator.simulate()
    schedule = simulator.gen_schedule_graph_no_comm()
    # print(schedule)
    simulator.plot()
    # simulator.export("./originalzb.txt")
    plt.show()
    # plt.savefig("test11.png")


if __name__ == '__main__':
    main()
