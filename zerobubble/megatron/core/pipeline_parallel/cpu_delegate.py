import torch
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp


class CommunicationDelegate(mp.Process):
    def __init__(self, name, msg_queue, task_queue, dist_info):
        super().__init__()
        self.role = 'sender' if 'Send' in name else 'recver'
        self.name = name
        self.msg_queue = msg_queue
        # if role == sender, then it pulls from the task_queue and send data to recver
        # if role == recver, then it recvs data and put results to the task_queue
        self.task_queue = task_queue
        self.dist_info = dist_info
        self.global_rank = self.dist_info['GLOBAL_RANK']
        if self.role == 'recver':
            assert 'data_shape' in dist_info and 'dtype' in dist_info
            self.recv_buffer = [
                torch.zeros(
                    self.dist_info['data_shape'],
                    requires_grad=False,  # It is enabled in cpu_schedules, when copy it to CUDA
                    device='cpu',
                    dtype=self.dist_info['dtype']
                ).share_memory_() for _ in range(20)
            ]
            self.recv_id = 0

    def run(self):
        delegate_rank = 0 if self.role == 'sender' else 1
        delegate_world_size = 2
        os.environ = {}
        os.environ['MASTER_ADDR'] = self.dist_info['MASTER_ADDR']
        os.environ['MASTER_PORT'] = str(self.dist_info['MASTER_PORT'])
        # os.environ['GLOO_SOCKET_IFNAME'] = self.dist_info['IFNAME']
        dist.init_process_group(
            backend='gloo',
            init_method="env://",
            rank=delegate_rank,
            world_size=delegate_world_size
        )
        # Init notification to main process
        # print(f"[{self.name} {self.global_rank}] Initialized with rank {delegate_rank} out of {delegate_world_size}")
        self.msg_queue.put("init")
        while True:
            time.sleep(0.1)
            iter_task_mbs = self.msg_queue.get()
            if iter_task_mbs == 'init':
                self.msg_queue.put("init")
            else:
                break

        if self.role == 'sender':
            while True:
                # print(f"[{self.name} {self.global_rank}] Current iteration contains {iter_task_mbs} microbatches to send.")
                if iter_task_mbs is None:
                    # print(f"[{self.name} {self.global_rank}] Received sentinel. Exiting.")
                    break
                else:
                    for i in range(int(iter_task_mbs)):
                        task = self.task_queue.get()
                        if task is None:
                            continue
                        assert isinstance(task, torch.Tensor)
                        # print(f"[{self.name} {self.global_rank}] Get task {i}: {task}.")
                        dist.send(task, dst=1)
                        # print(f"[{self.name} {self.global_rank}] Successfully sent microbatch {i}.")
                iter_task_mbs = self.msg_queue.get()

        elif self.role == 'recver':
            while True:
                # print(f"[{self.name} {self.global_rank}] Current iteration contains {iter_task_mbs} microbatches to receive.")
                self.recv_id = 0
                if iter_task_mbs is None:
                    # print(f"[{self.name} {self.global_rank}] Received sentinel. Exiting.")
                    break
                else:
                    for i in range(int(iter_task_mbs)):
                        # print(f"[{self.name} {self.global_rank}] Before recv {i}: buffershape: {self.recv_buffer[i].shape}.")
                        dist.recv(self.recv_buffer[i], src=0)
                        # print(f"[{self.name} {self.global_rank}] Successfully received microbatch {i}.")
                        self.task_queue.put(self.recv_buffer[i])
                iter_task_mbs = self.msg_queue.get()
        dist.destroy_process_group()
        # print(f"[{self.name} {self.global_rank}] Terminated!")


def start_communication_delegate(name, msg_queue, task_queue, dist_info):
    delegate = CommunicationDelegate(name, msg_queue, task_queue, dist_info)
    delegate.start()
    return delegate


class DelegateManager():
    def __init__(self, my_stage, total_stages, my_addr, prev_addr, next_addr, dele_dshape, dele_dtype, num_delegates):
        '''
        Role(MASTER_ADDR, MASTER_PORT)
        Stage0: SF(0, 12306), SB(None), RF(None), RB(1, 12317)
        Stage1: SF(1, 12306), SB(1, 12317), RF(0, 12306), RB(2, 12317)
        Stage2: SF(2, 12306), SB(2, 12317), RF(1, 12306), RB(3, 12317)
        Stage3: SF(None), SB(3, 12317), RF(2, 12306), RB(None)
        '''
        # print("[DelegateManager] Init start.")
        mp.set_start_method('spawn', force=True)
        dist_info = {"GLOBAL_RANK": dist.get_rank(), 'MASTER_ADDR': None, "MASTER_PORT": 12306, 'data_shape': dele_dshape, 'dtype': dele_dtype}
        self.total_stages = total_stages
        self.my_stage = my_stage
        self.num_delegates = num_delegates
        self.queues = {}
        self.delegates = {}
        # (1) Create: send to the next node (send_forward)
        if my_stage != total_stages - 1:  # not the last PP stage
            for qid in range(self.num_delegates):
                dist_info_send_forward = dist_info.copy()
                dist_info_send_forward['MASTER_ADDR'] = my_addr
                dist_info_send_forward['MASTER_PORT'] = 12306 + qid
                self.queues[f'send_forward_msg_{qid}'] = mp.Queue()
                self.queues[f'send_forward_task_{qid}'] = mp.Queue()
                self.delegates[f'send_forward_{qid}'] = start_communication_delegate(f'SendForward{qid}', self.queues[f'send_forward_msg_{qid}'], self.queues[f'send_forward_task_{qid}'], dist_info_send_forward)

        # (2) Create: send to the prev node (send_backward)
        if my_stage != 0:  # not the first PP stage
            for qid in range(self.num_delegates):
                dist_info_send_backward = dist_info.copy()
                dist_info_send_backward['MASTER_ADDR'] = my_addr
                dist_info_send_backward['MASTER_PORT'] = 12317 + qid
                self.queues[f'send_backward_msg_{qid}'] = mp.Queue()
                self.queues[f'send_backward_task_{qid}'] = mp.Queue()
                self.delegates[f'send_backward_{qid}'] = start_communication_delegate(f'SendBackward{qid}', self.queues[f'send_backward_msg_{qid}'], self.queues[f'send_backward_task_{qid}'], dist_info_send_backward)

        # (3) Create: recv from the prev node (recv_forward)
        if my_stage != 0:  # not the first PP stage
            for qid in range(self.num_delegates):
                dist_info_recv_forward = dist_info.copy()
                dist_info_recv_forward['MASTER_ADDR'] = prev_addr
                dist_info_recv_forward['MASTER_PORT'] = 12306 + qid
                self.queues[f'recv_forward_msg_{qid}'] = mp.Queue()
                self.queues[f'recv_forward_task_{qid}'] = mp.Queue()
                self.delegates[f'recv_forward_{qid}'] = start_communication_delegate(f'RecvForward{qid}', self.queues[f'recv_forward_msg_{qid}'], self.queues[f'recv_forward_task_{qid}'], dist_info_recv_forward)

        # (4) Create: recv from the next node (recv_backward)
        if my_stage != total_stages - 1:  # not the last PP stage
            for qid in range(self.num_delegates):
                dist_info_recv_backward = dist_info.copy()
                dist_info_recv_backward['MASTER_ADDR'] = next_addr
                dist_info_recv_backward['MASTER_PORT'] = 12317 + qid
                self.queues[f'recv_backward_msg_{qid}'] = mp.Queue()
                self.queues[f'recv_backward_task_{qid}'] = mp.Queue()
                self.delegates[f'recv_backward_{qid}'] = start_communication_delegate(f'RecvBackward{qid}', self.queues[f'recv_backward_msg_{qid}'], self.queues[f'recv_backward_task_{qid}'], dist_info_recv_backward)

        # print("[DelegateManager] Init Wait.")
        # Wait for initialization
        for i in range(self.num_delegates):
            if my_stage != total_stages - 1:
                assert self.queues[f'send_forward_msg_{i}'].get() == 'init'
            if my_stage != 0:
                assert self.queues[f'send_backward_msg_{i}'].get() == 'init'
            if my_stage != 0:
                assert self.queues[f'recv_forward_msg_{i}'].get() == 'init'
            if my_stage != total_stages - 1:
                assert self.queues[f'recv_backward_msg_{i}'].get() == 'init'
        # print("[DelegateManager] Init Done.")

    def start_iter(self, num_mbs):
        assert num_mbs % self.num_delegates == 0
        # Round robin assignment
        num_mbs = num_mbs // self.num_delegates
        for i in range(self.num_delegates):
            if self.my_stage != self.total_stages - 1:
                self.queues[f'send_forward_msg_{i}'].put(num_mbs)
            if self.my_stage != 0:
                self.queues[f'send_backward_msg_{i}'].put(num_mbs)
            if self.my_stage != 0:
                self.queues[f'recv_forward_msg_{i}'].put(num_mbs)
            if self.my_stage != self.total_stages - 1:
                self.queues[f'recv_backward_msg_{i}'].put(num_mbs)

    def terminate(self):
        for i in range(self.num_delegates):
            if self.my_stage != self.total_stages - 1:
                self.queues[f'send_forward_msg_{i}'].put(None)
                self.delegates[f'send_forward_{i}'].join()
            if self.my_stage != 0:
                self.queues[f'send_backward_msg_{i}'].put(None)
                self.delegates[f'send_backward_{i}'].join()
            if self.my_stage != 0:
                self.queues[f'recv_forward_msg_{i}'].put(None)
                self.delegates[f'recv_forward_{i}'].join()
            if self.my_stage != self.total_stages - 1:
                self.queues[f'recv_backward_msg_{i}'].put(None)
                self.delegates[f'recv_backward_{i}'].join()
        print("[DelegateManager] Shutdown gracefully.")


if __name__ == '__main__':
    # Test delegate
    dist.init_process_group(backend='gloo', init_method="env://")
    my_rank = dist.get_rank()
    world_size = dist.get_world_size()
    shape = (5, 5)
    dtype = torch.float32
    # print(f"[Rank{my_rank}] Inited!")

    ip_list = ["172.24.82.221", "172.24.82.222", "172.24.82.223", "172.24.82.224"]
    my_ip = ip_list[my_rank]
    prev_ip = ip_list[my_rank - 1] if my_rank != 0 else None
    next_ip = ip_list[my_rank + 1] if my_rank != world_size - 1 else None
    manager = DelegateManager(my_rank, world_size, my_ip, prev_ip, next_ip, shape, dtype)

    def f(x):
        return x + 1

    def b(x):
        return x - 1

    def w(x):
        assert torch.sum(x) == 0

    def sched_f():
        if my_rank == 0:
            x = torch.zeros((5, 5), dtype=torch.float32)
        else:
            x = manager.recv_forward_task_queue.get()
        # print(f"[Rank{my_rank}] F: x={x}!")
        time.sleep(0.03)
        x = f(x)
        if my_rank != world_size - 1:
            manager.send_forward_task_queue.put(x, block=False)

    def sched_b():
        if my_rank == world_size - 1:
            x = torch.full((5, 5), world_size, dtype=torch.float32)
        else:
            x = manager.recv_backward_task_queue.get()
        # print(f"[Rank{my_rank}] B: x={x}!")
        time.sleep(0.03)
        x = b(x)
        if my_rank != 0:
            manager.send_backward_task_queue.put(x, block=False)

    def sched_w():
        # print(f"[Rank{my_rank}] W!")
        time.sleep(0.03)

    for it in range(1):
        B = 12
        manager.start_iter(B)
        for mb in range(B):
            sched_f()
        for mb in range(B):
            sched_b()
        for mb in range(B):
            sched_w()
        dist.barrier()

    manager.terminate()
    # print(f"[Rank{my_rank}] Done!")
    dist.destroy_process_group()
