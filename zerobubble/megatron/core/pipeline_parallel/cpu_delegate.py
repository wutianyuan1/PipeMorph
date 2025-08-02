import torch
import time
import os
import redis
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import ctypes
import memcpy
from multiprocessing.shared_memory import SharedMemory
from megatron.core import parallel_state

CHECK_INTERVAL = 8
FLOAT16_NBYTES = 2
UNREADY_SIGNAL = 0
SEND_SIGNAL = 1
RECV_SIGNAL_CPU = 2
RECV_SIGNAL_GPU = 3
TP_MAXSIZE = 16
DP_MAXSIZE = 16


def addr_of(buffer, offset):
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer, offset))

def get_shm_signal(buffer, index):
    return torch.from_numpy(np.ndarray((1,), np.float16, buffer[index * FLOAT16_NBYTES : (index + 1) * FLOAT16_NBYTES]))

def get_shm_tensor(buffer, index, num_microbatches, tensor_shape, tensor_numel):
    signals_size = num_microbatches * FLOAT16_NBYTES
    tensor_size = tensor_numel * FLOAT16_NBYTES
    return torch.from_numpy(np.ndarray(tensor_shape, np.float16, buffer[signals_size + index * tensor_size : signals_size + (index + 1) * tensor_size]))


class CommunicationDelegate(mp.Process):
    def __init__(self, name, msg_queue, task_queue, shm, dist_info):
        super().__init__()
        self.role = 'sender' if 'Send' in name else 'recver'
        self.name = name
        self.msg_queue = msg_queue
        # if role == sender, then it pulls from the task_queue and send data to recver
        # if role == recver, then it recvs data and put results to the task_queue
        self.task_queue = task_queue
        self.shm = shm
        self.index = 0
        self.num_elements = torch.prod(torch.tensor((dist_info['data_shape']))).item()
        self.dist_info = dist_info
        self.pp_stage = self.dist_info['pp_stage']
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

        # Delay simulation
        self.delay_time = []
        self.delay_links = []
        self.delay_time_cache = None
        self.send_count = 0

    def simulate_delay(self):
        # Only sender calls this function!
        assert self.role == 'sender'
        # Update delay and slow links every CHECK_INTERVAL sends
        if self.send_count == 0:
            self.delay_links = []
            self.delay_time = []
            delay_links_reply = self.redis_client.get("slow_links")
            if delay_links_reply is not None:
                delay_links_str = delay_links_reply.decode()
                if delay_links_str != "":
                    for pair in delay_links_str.split(","):
                        start, end = pair.split("_")
                        self.delay_links.append((int(start), int(end)))
            delay_time_reply = self.redis_client.get("sleep_time")
            if delay_time_reply is not None:
                delay_time_str = delay_time_reply.decode()
                if delay_time_str != "":
                    delay_time = [float(t) for t in delay_time_str.split(',')]
                    self.delay_time = delay_time
            assert len(self.delay_time) == len(self.delay_links)
            # print(f"[{self.name} {self.pp_stage}] Update delay: links={self.delay_links}, time={self.delay_time}")

            self.delay_time_cache = 0
            for link, delay in zip(self.delay_links, self.delay_time):
                if ('SendForward' in self.name) and (self.pp_stage == link[0]):
                    self.delay_time_cache = delay
                    break
                elif ('SendBackward' in self.name) and (self.pp_stage == link[1]):
                    self.delay_time_cache = delay
                    break
            # print(f"[{self.name} {self.pp_stage}] Delay time init to {self.delay_time_cache}")

        self.send_count = (self.send_count + 1) % CHECK_INTERVAL

        if self.delay_time_cache != 0:
            time.sleep(self.delay_time_cache)

    def run(self):
        delegate_rank = 0 if self.role == 'sender' else 1
        delegate_world_size = 2
        redis_master_addr = os.environ['MASTER_ADDR']
        redis_port = os.environ.get('REDIS_PORT', '6379')
        os.environ = {}
        os.environ['MASTER_ADDR'] = self.dist_info['MASTER_ADDR']
        os.environ['MASTER_PORT'] = str(self.dist_info['MASTER_PORT'])
        os.environ['REDIS_MASTER_ADDR'] = redis_master_addr
        os.environ['REDIS_PORT'] = redis_port
        # os.environ['GLOO_SOCKET_IFNAME'] = self.dist_info['IFNAME']
        dist.init_process_group(
            backend='gloo',
            init_method=f"tcp://{self.dist_info['MASTER_ADDR']}:{self.dist_info['MASTER_PORT']}",
            rank=delegate_rank,
            world_size=delegate_world_size
        )
        self.redis_client = redis.StrictRedis(host=os.environ['REDIS_MASTER_ADDR'], port=int(os.environ.get('REDIS_PORT', '6379')))
        # Init notification to main process
        # print(f"[{self.name} {self.pp_stage}] Initialized with rank {delegate_rank} out of {delegate_world_size}")
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
                # print(f"[{self.name} {self.pp_stage}] Current iteration contains {iter_task_mbs} microbatches to send.")
                if iter_task_mbs is None:
                    # print(f"[{self.name} {self.pp_stage}] Received sentinel. Exiting.")
                    break
                else:
                    for i in range(int(iter_task_mbs)):
                        if self.dist_info['ipc_way']['send_way'] == 'queue':
                            tensor = self.task_queue.get()
                            assert isinstance(tensor, torch.Tensor)
                            # print(f"[{self.name} {self.pp_stage}] Get task {i}: {task}.")
                            self.simulate_delay()
                            dist.send(tensor, dst=1)
                            # print(f"[{self.name} {self.pp_stage}] Successfully sent microbatch {i}.")
                        else:
                            while True:
                                signal = get_shm_signal(self.shm.buf, i)
                                if signal.item() == SEND_SIGNAL:
                                    signal[0] = UNREADY_SIGNAL
                                    tensor = get_shm_tensor(self.shm.buf, i, self.dist_info['num_mb'], self.dist_info['data_shape'], self.num_elements)
                                    break
                            assert isinstance(tensor, torch.Tensor)
                            self.simulate_delay()
                            dist.send(tensor, dst=1)
                iter_task_mbs = self.msg_queue.get()

        elif self.role == 'recver':
            while True:
                # print(f"[{self.name} {self.pp_stage}] Current iteration contains {iter_task_mbs} microbatches to receive.")
                self.recv_id = 0
                if iter_task_mbs is None:
                    # print(f"[{self.name} {self.pp_stage}] Received sentinel. Exiting.")
                    break
                else:
                    for i in range(int(iter_task_mbs)):
                        if self.dist_info['ipc_way']['recv_way'] == 'queue':
                            # print(f"[{self.name} {self.pp_stage}] Before recv {i}: buffershape: {self.recv_buffer[i].shape}.")
                            dist.recv(self.recv_buffer[i], src=0)
                            # print(f"[{self.name} {self.pp_stage}] Successfully received microbatch {i}.")
                            self.task_queue.put(self.recv_buffer[i])
                        else:
                            tensor = get_shm_tensor(self.shm.buf, i, self.dist_info['num_mb'], self.dist_info['data_shape'], self.num_elements)
                            dist.recv(tensor, src=0)
                            signal = get_shm_signal(self.shm.buf, i)
                            signal[0] = RECV_SIGNAL_CPU
                iter_task_mbs = self.msg_queue.get()
        dist.destroy_process_group()
        # print(f"[{self.name} {self.pp_stage}] Terminated!")


def start_communication_delegate(name, msg_queue, task_queue, shm, dist_info):
    delegate = CommunicationDelegate(name, msg_queue, task_queue, shm, dist_info)
    delegate.start()
    return delegate


class DelegateManager():
    def __init__(self, my_stage, total_stages, my_addr, prev_addr, next_addr, dele_dshape, dele_dtype, num_delegates, ipc_way, num_microbatches):
        '''
        Role(MASTER_ADDR, MASTER_PORT)
        Stage0: SF(0, 12306), SB(None), RF(None), RB(1, 23306)
        Stage1: SF(1, 12306), SB(1, 23306), RF(0, 12306), RB(2, 23306)
        Stage2: SF(2, 12306), SB(2, 23306), RF(1, 12306), RB(3, 23306)
        Stage3: SF(None), SB(3, 23306), RF(2, 12306), RB(None)
        '''
        # print("[DelegateManager] Init start.")
        mp.set_start_method('spawn', force=True)
        dist_info = {'MASTER_ADDR': None, "MASTER_PORT": 12306, "pp_stage": parallel_state.get_pipeline_model_parallel_rank(), 'data_shape': dele_dshape, 'dtype': dele_dtype, 'ipc_way': ipc_way, 'num_mb': num_microbatches}
        self.total_stages = total_stages
        self.my_stage = my_stage
        self.num_delegates = num_delegates
        self.queues = {}
        self.shms = {}
        self.delegates = {}
        tp_dp_idx = parallel_state.get_data_parallel_rank() * TP_MAXSIZE * DP_MAXSIZE + parallel_state.get_tensor_model_parallel_rank() * TP_MAXSIZE
        rank = dist.get_rank()
        # (1) Create: send to the next node (send_forward)
        if my_stage != total_stages - 1:  # not the last PP stage
            for qid in range(self.num_delegates):
                dist_info_send_forward = dist_info.copy()
                dist_info_send_forward['MASTER_ADDR'] = my_addr
                dist_info_send_forward['MASTER_PORT'] = 12306 + tp_dp_idx + qid
                self.queues[f'send_forward_msg_{qid}'] = mp.Queue()
                self.queues[f'send_forward_task_{qid}'] = mp.Queue()
                self.shms[f'send_forward_shm_{qid}'] = SharedMemory(name=f'send_forward_shm_{qid}_{rank}', create=True, size=2 * num_microbatches * (torch.prod(torch.tensor(dele_dshape)).item() + 1))
                memcpy.register_pinned_memory(addr_of(self.shms[f'send_forward_shm_{qid}'].buf, 0), self.shms[f'send_forward_shm_{qid}'].size)
                self.delegates[f'send_forward_{qid}'] = start_communication_delegate(f'SendForward{qid}', self.queues[f'send_forward_msg_{qid}'], self.queues[f'send_forward_task_{qid}'], self.shms[f'send_forward_shm_{qid}'], dist_info_send_forward)

        # (2) Create: send to the prev node (send_backward)
        if my_stage != 0:  # not the first PP stage
            for qid in range(self.num_delegates):
                dist_info_send_backward = dist_info.copy()
                dist_info_send_backward['MASTER_ADDR'] = my_addr
                dist_info_send_backward['MASTER_PORT'] = 23306 + tp_dp_idx + qid
                self.queues[f'send_backward_msg_{qid}'] = mp.Queue()
                self.queues[f'send_backward_task_{qid}'] = mp.Queue()
                self.shms[f'send_backward_shm_{qid}'] = SharedMemory(name=f'send_backward_shm_{qid}_{rank}', create=True, size=2 * num_microbatches * (torch.prod(torch.tensor(dele_dshape)).item() + 1))
                memcpy.register_pinned_memory(addr_of(self.shms[f'send_backward_shm_{qid}'].buf, 0), self.shms[f'send_backward_shm_{qid}'].size)
                self.delegates[f'send_backward_{qid}'] = start_communication_delegate(f'SendBackward{qid}', self.queues[f'send_backward_msg_{qid}'], self.queues[f'send_backward_task_{qid}'], self.shms[f'send_backward_shm_{qid}'], dist_info_send_backward)

        # (3) Create: recv from the prev node (recv_forward)
        if my_stage != 0:  # not the first PP stage
            for qid in range(self.num_delegates):
                dist_info_recv_forward = dist_info.copy()
                dist_info_recv_forward['MASTER_ADDR'] = prev_addr
                dist_info_recv_forward['MASTER_PORT'] = 12306 + tp_dp_idx + qid
                self.queues[f'recv_forward_msg_{qid}'] = mp.Queue()
                self.queues[f'recv_forward_task_{qid}'] = mp.Queue()
                self.shms[f'recv_forward_shm_{qid}'] = SharedMemory(name=f'recv_forward_shm_{qid}_{rank}', create=True, size=2 * num_microbatches * (torch.prod(torch.tensor(dele_dshape)).item() + 1))
                memcpy.register_pinned_memory(addr_of(self.shms[f'recv_forward_shm_{qid}'].buf, 0), self.shms[f'recv_forward_shm_{qid}'].size)
                self.delegates[f'recv_forward_{qid}'] = start_communication_delegate(f'RecvForward{qid}', self.queues[f'recv_forward_msg_{qid}'], self.queues[f'recv_forward_task_{qid}'], self.shms[f'recv_forward_shm_{qid}'], dist_info_recv_forward)

        # (4) Create: recv from the next node (recv_backward)
        if my_stage != total_stages - 1:  # not the last PP stage
            for qid in range(self.num_delegates):
                dist_info_recv_backward = dist_info.copy()
                dist_info_recv_backward['MASTER_ADDR'] = next_addr
                dist_info_recv_backward['MASTER_PORT'] = 23306 + tp_dp_idx + qid
                self.queues[f'recv_backward_msg_{qid}'] = mp.Queue()
                self.queues[f'recv_backward_task_{qid}'] = mp.Queue()
                self.shms[f'recv_backward_shm_{qid}'] = SharedMemory(name=f'recv_backward_shm_{qid}_{rank}', create=True, size=2 * num_microbatches * (torch.prod(torch.tensor(dele_dshape)).item() + 1))
                memcpy.register_pinned_memory(addr_of(self.shms[f'recv_backward_shm_{qid}'].buf, 0), self.shms[f'recv_backward_shm_{qid}'].size)
                self.delegates[f'recv_backward_{qid}'] = start_communication_delegate(f'RecvBackward{qid}', self.queues[f'recv_backward_msg_{qid}'], self.queues[f'recv_backward_task_{qid}'], self.shms[f'recv_backward_shm_{qid}'], dist_info_recv_backward)

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
        for shm in self.shms.values():
            if shm is not None:
                # reset all signals
                for i in range(num_mbs):
                    signal = torch.from_numpy(np.ndarray((1,), np.float16, shm.buf[i * 2 : (i + 1) * 2]))
                    signal[0] = 0
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
        for shm in self.shms.values():
            if shm is not None:
                shm.unlink()
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
