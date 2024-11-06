from collections import deque
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
        # TODO fix
        self.mini_batches_to_fwd = [[] for _ in range(depth)]
        self.mini_batches_finish_bi=[[] for _ in range(depth)]
    def init_queues(self):
        for i in range(self.pipeline_depth):
            self.mini_batches_to_fwd[i] = [j for j in range(1, num_mini+1)]
        for i in range(1, min(self.max_mini_batch_num, self.num_mini)+1):
            self.fwd_queues[0].append(i)
        print("self.mini_batches_to_fwd")
        print(self.mini_batches_to_fwd)
            
    def pick_queue(self, stage, time, all_tasks):
        if len(self.bi_queues[stage]) > 0:
            if stage != self.pipeline_depth -1 and self.bi_queues[stage][0] not in self.mini_batches_finish_bi[stage+1]:
                pass
            else:
                return self.bi_queues[stage], 'bi'
        if len(self.fwd_queues[stage]) > 0:
            if len(self.bw_queues[stage]) >= self.max_mini_batch_num:
                return self.bw_queues[stage], 'bw'
            else:
                if stage != 0 and self.fwd_queues[stage][0]  in self.mini_batches_to_fwd[stage-1]:
                    pass
                else:
                    return self.fwd_queues[stage], 'fwd'
        
        
        if len(self.bw_queues[stage]) > 0:
            return self.bw_queues[stage], 'bw'
        return None, None

    def print_task(self, all_tasks, num_stages):
        task_map = "FbB"
        maxt = 0
        pp_schedule = [['Z0' for _ in range(100)] for _ in range(num_stages)]
        
        for j in range(num_stages):
            tasks = all_tasks[j]
            for task in tasks:
                if task.task == 'fwd':
                    pp_schedule[j][task.time] = 'F' + str(task.batch_index)
                elif task.task == 'bi' or task.task =='bw':
                    pp_schedule[j][task.time] = 'B' + str(task.batch_index)
            
                maxt = max(maxt, task.time)
        
        for j in range(num_stages):
            print(f"Stage{j}: [{' '.join(pp_schedule[j][:maxt])}]")

    def generate(self):
        num_bubbles = [0] * self.pipeline_depth
        cc_nodes = [10, 20]
        time = 0
        all_tasks = [[] for _ in range(self.pipeline_depth)]
        
        while time < 20000:
            mini_batches = []
            queue_ids = []
            all_queues_empty = True
            
            for i in range(self.pipeline_depth):
                queue, identifier = self.pick_queue(i, time, all_tasks)
                # -1 就是bubble
                if i == 0 or i == 1:
                    print("self.bw",i,self.bw_queues[i])
                    print("self.bi",i,self.bi_queues[i])
                    print("self.fwd",i,self.fwd_queues[i])
                    print("self.mini_batches_to_fwd",i,self.mini_batches_to_fwd[i])
                    
                mini = queue.popleft() if queue is not None else -1
                mini_batches.append(mini)
                
                if mini ==-1:
                    num_bubbles[i] += 1
                    queue_ids.append('Z')
                    continue
                
                all_queues_empty = False
                queue_ids.append(identifier)
                
                # print("self.mini_batches_to_fwd[i].pop(0)",self.mini_batches_to_fwd)
                # print("queue",queue,identifier,i)
                if identifier == 'fwd':
                    # print("mini",i, mini)
                    
                    if len(self.mini_batches_to_fwd[i]) > 0 and mini == self.mini_batches_to_fwd[i][0]:
                        self.mini_batches_to_fwd[i].pop(0)
                if identifier == 'bi':
                    self.mini_batches_finish_bi[i].append(mini)
                all_tasks[i].append(ScheduleTask(mini, identifier, time))
            
            if all_queues_empty:
                break
            
            for i in range(pipeline_depth):
                mini = mini_batches[i]
                if mini ==-1:
                    continue
                
                if queue_ids[i] == 'fwd':
                    if i != pipeline_depth - 1 and mini not in self.fwd_queues[i+1]:
                        # 是拥塞的结点,而且下一个结点没有东西算
                        if i == cc_nodes[0] and (len(self.fwd_queues[i+1]) == 0 and len(self.bi_queues[i+1]) == 0 and len(self.bw_queues[i+1]) == 0):
                            # 插入一个bubble和一个mini
                            self.fwd_queues[i + 1].append(-1)
                            self.fwd_queues[i + 1].append(mini)
                        else:
                            self.fwd_queues[i+1].append(mini)
                    else:
                        self.bi_queues[i].append(mini)
                elif queue_ids[i] == 'bi':
                    # 做完bi可以做bw
                    if mini not in self.bw_queues[i]:
                        self.bw_queues[i].append(mini)
                    if i != 0 and mini not in self.bi_queues[i-1]: # not first stage
                        # 如果有cc，而且没有东西算：
                        if i == cc_nodes[1] and (len(self.fwd_queues[i+1]) == 0 and len(self.bi_queues[i+1]) == 0 and len(self.bw_queues[i+1]) == 0):
                            self.bi_queues[i-1].append(-1)
                            self.bi_queues[i-1].append(mini)
                        else:  
                            self.bi_queues[i-1].append(mini)
                    pass
                elif queue_ids[i] == 'bw':
                    # 如果还有batch没有做过forward那么把它加进来
                    if len(self.mini_batches_to_fwd[i]) > 0:
                        if i == 0:
                            self.fwd_queues[i].append(self.mini_batches_to_fwd[i][0])
                        elif (len(self.mini_batches_to_fwd[i-1]) >0 and self.mini_batches_to_fwd[i][0]<self.mini_batches_to_fwd[i-1][0]) or len(self.mini_batches_to_fwd[i-1])>0:
                            if self.mini_batches_to_fwd[i][0] not in self.fwd_queues[i]:
                                self.fwd_queues[i].append(self.mini_batches_to_fwd[i][0])
                            

                else:
                    print("Should never be here")
            
            time += 1

        for i in range(pipeline_depth):
            if i == self.rank:
                for task in all_tasks[i]:
                    print(f"{task.task}, {task.batch_index - 1};")
        self.print_task(all_tasks, 4)
        print("Fraction of bubbles = {}/{}  {:.2f} percent".format(num_bubbles[0]-1, time-1, (num_bubbles[0]-1)*100/(time-1)))

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
    s.init_queues()
    s.generate()