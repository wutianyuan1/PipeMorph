from enum import Enum, auto
from collections import deque
class Task:
    def __init__(self, stage, mini_batch, task):
        self.stage = stage
        self.batch_idx = mini_batch
        self.task = task # ['fwd', 'bi', 'bw']

class ScheduleTask:
    def __init__(self, index, task, time):
        self.batch_index = index
        self.task = task
        self.time = time
class GenSchedule:
    def __init__(self, depth, num_mini, rank, max_mini_batch_num):
        self.pipeline_depth = depth
        self.num_mini = num_mini
        self.rank = rank
        self.max_mini_batch_num = max_mini_batch_num
        self.fwd_queues = [deque() for _ in range(depth)]
        self.bi_queues = [deque() for _ in range(depth)]
        self.bw_queues = [deque() for _ in range(depth)]
        self.task_list = ['fwd', 'bi', "bw"]
        # TODO fix
        self.mini_batches_to_fwd = [[] for _ in range(depth)]
        self.mini_batches_finish_bi=[[] for _ in range(depth)]
    def add_dependency(self, task):
        if task.task == 'fwd':
            # fwd 需要当前stage 比自己index小的batch fwd，前一个stage minibatch fwd结束
            for i in range(task.mini_batch -1):
                self.graph[task].append(Task(task.stage, i+1, task.task))
            if task.stage != 0:
                for i in range(task.stage):
                    self.graph[task].append(Task(i, task.batch_idx, task.task))
        
        
                    
        pass
    # def get_idx(self, stage, mini_batch, task):
    #     idx = stage*(self.num_mini*len(self.task_list))+ (mini_batch-1) * len(self.task_list)+(self.task_list.index(task))
       
    #     return idx
    def init_graph(self):
        node_num = self.pipeline_depth * self.num_mini * 3 #三种类型的任务
        print("node_num", node_num)
        self.graph = {}
        for i in range(self.pipeline_depth):
            for j in range(1, self.num_mini+1):
                for t in self.task_list:
                    task = Task(i, j, t)
                    # self.get_idx(i, j, t)
                    # print("ii", )
                    self.graph[task] = []
        for task in self.graph:
            self.graph[task] = self.add_dependency(task)
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: gen-schedule <pipeline-depth> <num_micro_batches> <device_rank>")
        sys.exit(-1)
    
    pipeline_depth = int(sys.argv[1])
    num_mini = int(sys.argv[2])
    rank = int(sys.argv[3])
    max_mini_batch_num = 4
    s = GenSchedule(pipeline_depth, num_mini, rank, max_mini_batch_num)
    s.init_graph()
    # s.generate()