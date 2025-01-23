import contextlib
import itertools
from typing import Iterator, List, Union

import torch.distributed
from pipeline_simulator import auto_schedule
import copy
# from megatron.core.pipeline_parallel import auto_schedule
import ctypes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import torch
import time
import redis

from megatron import core, get_args, get_num_microbatches, print_rank_0
# from megatron.core.pipeline_parallel.ll import Node, LinkedList
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import v_schedule, v_schedule_greedy
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)
from megatron.core.weight_grad_store import WeightGradStore
from megatron.timers import Timer
from megatron.utils import is_second_last_pipeline_stage

CHECK_INTERVAL = 2
TIME_BUFFER_SIZE = 4
EXEC_TIME_PROFILE_END_ITER=-100
SCHEDULE_UPDATE_START_ITER=10000
AUTO_SCHEDULE_COMMUNICATION_TYPES = {'RECV_FORWARD', 'RECV_BACKWARD', 'SEND_FORWARD', 'SEND_BACKWARD'}


class ScheduleTimers:
    iter_counter = 0
    comm_time = 0
    concluded = False
    
    chunks = []

    def __init__(self):
        self.f = Timer('f')
        self.b = Timer('b')
        self.w = Timer('w')
        self.f_cnt = 0
        self.b_cnt = 0
        self.w_cnt = 0
        self.f_mem = 0
        self.b_mem = 0
    
    def conclusion(self):
        # assert self.concluded
        assert self.f_cnt > 0
        assert self.b_cnt > 0
        avg_f = int(self.f.elapsed(reset=False) / self.f_cnt * 1000000)
        avg_b = int(self.b.elapsed(reset=False) / self.b_cnt * 1000000)
        avg_f_mem = self.f_mem / self.f_cnt // 1000000
        avg_b_mem = self.b_mem / self.b_cnt // 1000000
        if self.w_cnt > 0:
            avg_w = int(self.w.elapsed(reset=False) / self.w_cnt * 1000000)
        else:
            avg_w = avg_b
        avg_w_mem = 0 - avg_f_mem - avg_b_mem
        return (avg_f, avg_b, avg_w, int(self.comm_time * 1000000), 
            avg_f_mem, avg_b_mem, avg_w_mem)

    @classmethod
    def for_chunk(cls, chunk):
        while len(cls.chunks) <= chunk:
            cls.chunks.append(cls())
        return cls.chunks[chunk]
    
    @classmethod
    def joint_conclusion(cls):
        ret = [x.conclusion() for x in cls.chunks]
        ret = list(zip(*ret))
        # C is shared bwteen chunks
        ret[3] = ret[3][0]
        return ret




def bootstrap_and_profile_p2p_communication(
    config, send_tensor_shapes, recv_tensor_shapes
    ):
    if ScheduleTimers.iter_counter == 1 and parallel_state.get_pipeline_model_parallel_world_size() > 1:
        nccl_init_tensor = [torch.Tensor([0]).cuda()]
        shape = [(1,)]
        if get_args().zero_bubble_v_schedule:
            # Make everyone think they are the first chunk, so we still need additional check to prevent rank -1 to send_forward/recv_backward
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_forward(shape, config)
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            send_forward(nccl_init_tensor, shape, config)
            recv_backward(shape, config)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            send_backward(nccl_init_tensor, shape, config)

        # Benchmarking the communication cost
        send_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in send_tensor_shapes]
        recv_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in recv_tensor_shapes]
        torch.distributed.barrier()
        t = Timer('comm-benchmark')
        t.start()
        print_rank_0(
            f"Start benchmarking communication with size {recv_tensor_shapes}, {send_tensor_shapes}")
        for i in range(10):
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_forward(recv_tensor_shapes, config)
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                send_forward(send_data, send_tensor_shapes, config)
                recv_backward(send_tensor_shapes, config)
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                send_backward(recv_data, recv_tensor_shapes, config)
        t.stop()
        per_communication = torch.cuda.FloatTensor([t.elapsed() / (
                parallel_state.get_pipeline_model_parallel_world_size() - 1) / 2 / 10])
        torch.distributed.all_reduce(per_communication,
                                     torch.distributed.ReduceOp.MAX)
        ScheduleTimers.comm_time = per_communication.item()
        print_rank_0(f"Communication time: {ScheduleTimers.comm_time}")


def fused_pipeline_ops(
    tensor_send_prev: List[torch.Tensor],
    tensor_recv_prev: List[torch.Tensor],
    tensor_send_next: List[torch.Tensor],
    tensor_recv_next: List[torch.Tensor],
    repeat_times: int,
    slow_links,
    stage: int,
):
    ops = []
    # sp_reqs, rp_reqs, sn_reqs, rn_reqs = [], [], [], []
    group = get_pipeline_model_parallel_group()
    for t in tensor_send_prev:
        # send_prev_op = torch.distributed.send(torch.stack([t] * repeat_times, dim=0) if stage in [n for (p, n) in slow_links] else t, get_pipeline_model_parallel_prev_rank(), group)
        # sp_reqs.append(send_prev_op)
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            torch.stack([t] * repeat_times, dim=0) if stage in [n for (p, n) in slow_links] else t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    for t in tensor_recv_prev:
        # recv_prev_op = torch.distributed.recv(t, get_pipeline_model_parallel_prev_rank(), group)
        # rp_reqs.append(recv_prev_op)
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    for t in tensor_send_next:
        # send_next_op = torch.distributed.send(torch.stack([t] * repeat_times, dim=0) if stage in [p for (p, n) in slow_links] else t, get_pipeline_model_parallel_next_rank(), group)
        # sn_reqs.append(send_next_op)
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            torch.stack([t] * repeat_times, dim=0) if stage in [p for (p, n) in slow_links] else t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    for t in tensor_recv_next:
        # recv_next_op = torch.distributed.recv(t, get_pipeline_model_parallel_next_rank(), group)
        # rn_reqs.append(recv_next_op)
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    # return sp_reqs, rp_reqs, sn_reqs, rn_reqs
    if len(ops) > 0:
        # start, end = start_a_timer()
        reqs = torch.distributed.batch_isend_irecv(ops)
        # reqs = ops
        # t = elapsed_time(start, end)
        return reqs, None
    else:
        reqs = []
    return reqs, None


def start_a_timer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream=torch.cuda.current_stream())
    return start, end

def elapsed_time(start: torch.cuda.Event, end: torch.cuda.Event):
    end.record(stream=torch.cuda.current_stream())
    end.synchronize()
    # torch.cuda.synchronize()
    return start.elapsed_time(end)


class MicroBatch:
    def __init__(self, minibatch: int, type: str):
        self.minibatch = minibatch
        self.type = type


class ZeroBubbleScheduler:

    def __init__(self):
        self._reset()

        self.schedules = None
        self.send_tensor_shapes = None
        self.recv_tensor_shapes = None
        self.config = None
        self.run_timer = None
        self.forward_step_func = None
        self.data_iterator = None
        self.model = None
        self.model_type = None
        self.num_microbatches = None
        self.collect_non_loss_data = None
        self.forward_only = None
        self.no_sync_context = None
        self.no_sync_func = None

        self.it = 0
        self.do_post_validation = False
        self.is_first_run = True
        self.optimizer = None

        self.stage = parallel_state.get_pipeline_model_parallel_rank()
        self.num_stages = parallel_state.get_pipeline_model_parallel_world_size()

        # For network delay simulation
        # self.client = redis.StrictRedis('localhost', port=6379, db=0)
        self.iter_cnt = 0
        # self.sleep_time = 0
        self.slow_links = [(0, 1)]

        # For execution and communication time profiling
        # Information format: [total seconds, times of exection or reception]
        # An entry: xxxx_time_buffer[0] = [iteration index, time information of this iteration]
        self.exec_time_buffer = [[]] * TIME_BUFFER_SIZE
        self.recv_time_buffer = [[]] * TIME_BUFFER_SIZE
        self.exec_time_buffer[0] = [None, {"F": [0, 0], "B": [0, 0], "W": [0, 0]}]
        self.recv_time_buffer[0] = [None, {"prev": [0, 0], "next": [0, 0]}]
        self.exec_time = {"F": -1, "B": -1, "W": -1}
        if self.stage == 0:
            self.recv_time = {"next": -1}
        elif self.stage == self.num_stages - 1:
            self.recv_time = {"prev": -1}
        else:
            self.recv_time = {"prev": -1, "next": -1}

        self.comm_stream = None
        self.comm_streams = [torch.cuda.Stream() for _ in range(10)]
        self.comp_stream = None
        # self.comp_stream = torch.cuda.Stream()

    def _free_buffers(self):
        self.input_tensors = []
        self.output_tensors = []
        # self.send_forward_buffer = LinkedList()
        # self.recv_forward_buffer = LinkedList()
        # self.send_backward_buffer = LinkedList()
        # self.recv_backward_buffer = LinkedList()
        self.send_forward_buffer = []
        self.recv_forward_buffer = []
        self.send_backward_buffer = []
        self.recv_backward_buffer = []
        self.forward_data_store = []
    
    def _update_delay_info(self):
        raw_sleep_time = self.client.get("sleep_time")
        if raw_sleep_time is not None:
            raw_sleep_time = raw_sleep_time.decode()
            print_rank_0(f"[SLEEP_TIME] {raw_sleep_time}")
            try:
                self.sleep_time = float(raw_sleep_time)
            except:
                pass
        else:
            self.sleep_time = 0
        raw_slow_links = self.client.get("slow_links")
        # raw_slow_links: "(src_dst,)+"
        if raw_slow_links is not None:
            try:
                raw_slow_links = raw_slow_links.decode()
                print_rank_0(f"[SLOW_SLNKS] {raw_slow_links}")
                self.slow_links = []
                for link in raw_slow_links.split(","):
                    self.slow_links.append([int(i) for i in link.split("_")])
            except:
                pass
        else:
            self.slow_links = []

    def _update_exec_time_buffer(self, task_type: str, start: torch.cuda.Event, end: torch.cuda.Event, cnt: int = 1):
        if self.iter_cnt <= EXEC_TIME_PROFILE_END_ITER:
            end.record(stream=torch.cuda.current_stream())
            end.synchronize()
            self.exec_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][1][task_type][0] += start.elapsed_time(end)
            self.exec_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][1][task_type][1] += cnt

    # def _update_recv_time(self, recv_from: str, start: torch.cuda.Event, end: torch.cuda.Event):
    #     end.record(stream=torch.cuda.current_stream())
    #     end.synchronize()
    #     self.recv_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][1][recv_from][0] += start.elapsed_time(end)
    #     self.recv_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][1][recv_from][1] += 1

    def _update_recv_time_buffer(self, recv_froms: list[str], elapsed_time: float):
        for recv_from in recv_froms:
            self.recv_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][1][recv_from][0] += elapsed_time
            self.recv_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][1][recv_from][1] += 1

    def _update_exec_time(self):
        for task_type in self.exec_time:
            t, cnt = 0, 0
            for entry in self.exec_time_buffer:
                if len(entry) > 0:
                    t += entry[1][task_type][0]
                    cnt += entry[1][task_type][1]
            self.exec_time[task_type] = t / cnt

    def _update_recv_time(self):
        for recv_from in self.recv_time:
            t, cnt = 0, 0
            for entry in self.recv_time_buffer:
                if len(entry) > 0:
                    t += entry[1][recv_from][0]
                    cnt += entry[1][recv_from][1]
            if cnt != 0:
                self.recv_time[recv_from] = t / cnt

    def _sleep_before_recv(self, recv_from_prev: bool):
        for (prev, next) in self.slow_links:
            if (recv_from_prev and self.stage == next) or (not recv_from_prev and self.stage == prev):
                print("sleep", self.stage, self.sleep_time)
                # time.sleep(0.1)
                time.sleep(self.sleep_time)
                break

    def _update_schedule(self):
        while True:
            raw_schedule = self.client.get("schedule")
            if raw_schedule is not None:
                # TODO

                orders = {
                    # '1f1b': ["f f f f       b w f b w f b w f b w f b w f b w f b w f b w f b w   b w   b w   b w",
                    #            "f f f     b w f b w f b w f b w f b w f b w f b w f b w f b w f b w   b w   b w",
                    #              "f f   b w f b w f b w f b w f b w f b w f b w f b w f b w f b w f b w   b w",
                    #                "f b w f b w f b w f b w f b w f b w f b w f b w f b w f b w f b w f b w"],
                    # 'GPipe': ["f f f f f f             b b b b b b w w w w w w f f f f f f             b b b b b b w w w w w w",
                    #             "f f f f f f         b b b b b b w w w w w w     f f f f f f         b b b b b b w w w w w w",
                    #               "f f f f f f     b b b b b b w w w w w w         f f f f f f     b b b b b b w w w w w w",
                    #                 "f f f f f f b b b b b b w w w w w w             f f f f f f b b b b b b w w w w w w"],
                    # 'Manual': ["f f f f       b w f b w f b w   b w   b w   b w f f f f f f             b b b b b b w w w w w w",
                    #              "f f f     b w f b w f b w f b w   b w   b w     f f f f f f         b b b b b b w w w w w w",
                    #                "f f   b w f b w f b w f b w f b w   b w         f f f f f f     b b b b b b w w w w w w",
                    #                  "f b w f b w f b w f b w f b w f b w             f f f f f f b b b b b b w w w w w w"],
                    # 'GPipe': ["f f f f f f f f f f f f             b b b b b b b b b b b b w w w w w w w w w w w w",
                    #             "f f f f f f f f f f f f         b b b b b b b b b b b b w w w w w w w w w w w w",
                    #               "f f f f f f f f f f f f     b b b b b b b b b b b b w w w w w w w w w w w w",
                    #                 "f f f f f f f f f f f f b b b b b b b b b b b b w w w w w w w w w w w w"],
                    # 'ZeroBubble': [
                    #     "f f f f f f f b f b f b f b f b f b w b w b w b w b w b w b w w w w w w",
                    #       "f f f f f b f b f b f b f b f b f b f b w b w b w b w b w w w w w w w w",
                    #         "f f f b f b f b f b f b f b f b f b f b f b w b w b w w w w w w w w w w",
                    #           "f b f b f b f b f b f b f b f b f b f b f b f b w w w w w w w w w w w w"],
                    # 'ZeroBubble': [
                    #     "f f f f f f f b f b w f b b w f b b w f b b w f w w w w b w b w b w b w",
                    #       "f f f f f b f b f b f b w f b b w f b b w f b w f b w w b w w b w w w w",
                    #         "f f f b f b f b f b f b f b w f b b w f b w f b w f b w w b w w w w w w",
                    #           "f b f b f b f b f b f b f b f b w f b w f b w f b w f b w w w w w w w w"],

                    # 'Zb': [['F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'W'], ['RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'W', 'W', 'W', 'W'], ['RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'], ['RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']],
                    # Rank0 ['F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'RECV_BACKWARD', 'W', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'B', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'B', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'B', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'W', 'W', 'RECV_BACKWARD', 'W', 'W', 'B', 'RECV_BACKWARD', 'W', 'B', 'W', 'RECV_BACKWARD', 'B', 'W', 'B', 'W']
                    # Rank1 ['RECV_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'F', 'RECV_FORWARD', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_BACKWARD', 'W', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'W', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'W', 'B', 'SEND_BACKWARD', 'W', 'W', 'W', 'W']
                    # Rank2 ['RECV_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'F', 'RECV_FORWARD', 'SEND_FORWARD', 'RECV_FORWARD', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_BACKWARD', 'W', 'F', 'SEND_FORWARD', 'RECV_BACKWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'W', 'RECV_BACKWARD', 'F', 'SEND_FORWARD', 'B', 'SEND_BACKWARD', 'W', 'RECV_BACKWARD', 'W', 'B', 'SEND_BACKWARD', 'W', 'W', 'W', 'W', 'W', 'W']
                    # Rank3 ['RECV_FORWARD', 'RECV_FORWARD', 'F', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'F', 'B', 'SEND_BACKWARD', 'F', 'RECV_FORWARD', 'B', 'SEND_BACKWARD', 'W', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'W', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'W', 'F', 'B', 'SEND_BACKWARD', 'RECV_FORWARD', 'W', 'F', 'B', 'SEND_BACKWARD', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']
                    
                    # 'For0->1Slow': [
                    #     "f f f f f f f f f f f f   b w b w b w b w b w b w b w b w b w b w b w b w",
                    #             "f f f f f b f b f b f b f b f b f b f b w b w b w b w b w w w w w w w w",
                    #               "f f f b f b f b f b f b f b f b f b f b f b w b w b w w w w w w w w w w",
                    #                 "f b f b f b f b f b f b f b f b f b f b f b f b w w w w w w w w w w w w"],
                    # # [Rank0 TODO] F8
                    # # [Rank1 TODO] F9
                    # # [Rank2 TODO] F8
                    # # [Rank3 TODO] F3
                    # 'For1->2Slow': [
                    #     "f f f f f f f f f f f f   b w b w b w b w b w b w b w b w b w b w b w b w",
                    #       "f f f f f f f f f f f b f b w b w b w b w b w b w b w b w b w b w b w w",
                    #               "f f f b f b f b f b f b f b f b f b f b f b w b w b w w w w w w w w w w",
                    #                 "f b f b f b f b f b f b f b f b f b f b f b f b w w w w w w w w w w w w"],
                    # 'For2->3Slow': [
                    #     "f f f f f f f f f f f f   b w b w b w b w b w b w b w b w b w b w b w b w",
                    #       "f f f f f f f f f f f b f b w b w b w b w b w b w b w b w b w b w b w w",
                    #         "f f f f f f f f f b f b f b f b w b w b w b w b w b w b w b w b w w w w",
                    #                 "f b f b f b f b f b f b f b f b f b f b f b f b w w w w w w w w w w w w"],
                }
                # policy = '1f1b'
                # policy = 'GPipe'
                policy = list(orders.keys())[(self.iter_cnt - SCHEDULE_UPDATE_START_ITER) % len(orders)]
                schedules = [[] for _ in range(self.num_stages)]



                for i in range(self.num_stages):
                    minibatch = {"F": 0, "B": 0, "W": 0}
                    # for mb in orders[policy][i].split():
                    for mb in orders[policy][i]:
                        # if mb == 'f' and i != 0:
                        #     schedules[i].append(MicroBatch(minibatch[mb], 'RECV_FORWARD'))
                        # if mb == 'b' and i != self.num_stages - 1:
                        #     schedules[i].append(MicroBatch(minibatch[mb], 'RECV_BACKWARD'))
                        if mb in minibatch:
                            schedules[i].append(MicroBatch(minibatch[mb], mb.upper()))
                            minibatch[mb] += 1
                        else:
                            schedules[i].append(MicroBatch(None, mb.upper()))
                        # if mb == 'f' and i != self.num_stages - 1:
                        #     schedules[i].append(MicroBatch(minibatch[mb], 'SEND_FORWARD'))
                        # if mb == 'b' and i != 0:
                        #     schedules[i].append(MicroBatch(minibatch[mb], 'SEND_BACKWARD'))
                self.schedules = schedules[self.stage]
                print_rank_0(f"New schedules: {policy}")
                break

    def _reset(self):
        # Input, output tensors only need to be saved when doing backward passes
        self._free_buffers()
        self.send_handles = []
        self.communication_batch = {
            'SEND_NEXT': [],
            'RECV_NEXT': [],
            'SEND_PREV': [],
            'RECV_PREV': [],
        }

    @classmethod
    def direction_map(cls, node):
        return {
            'SEND_FORWARD': 'SEND_NEXT',
            'RECV_FORWARD': 'RECV_PREV',
            'SEND_BACKWARD': 'SEND_PREV',
            'RECV_BACKWARD': 'RECV_NEXT',
        }[node.type]

    def buffer_map(self, node):
        return {
            'SEND_FORWARD': self.send_forward_buffer,
            'RECV_FORWARD': self.recv_forward_buffer,
            'SEND_BACKWARD': self.send_backward_buffer,
            'RECV_BACKWARD': self.recv_backward_buffer,
        }[node.type]

    def flush(self, repeat_times: int = 1):
        name = '_'.join(
            [f'{v[0].type}.{v[0].minibatch}' for v in itertools.chain(
                *[vs for k, vs in self.communication_batch.items()])])
        assert self.send_tensor_shapes == self.recv_tensor_shapes
        assert len(self.send_tensor_shapes) == 1
        sn_tensors = [
            self.buffer_map(x[0]).pop(0)[0]
            for x in self.communication_batch['SEND_NEXT']
        ]
        sp_tensors = [
            self.buffer_map(x[0]).pop(0)[0]
            for x in self.communication_batch['SEND_PREV']
        ]
        if self.stage in [p for (p, n) in self.slow_links]:
            shapes = [repeat_times] + list(self.send_tensor_shapes[0])
        else:
            shapes = self.send_tensor_shapes[0]
        rn_tensors = [
            torch.zeros(
                shapes,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_NEXT']
        ]
        if self.stage in [n for (p, n) in self.slow_links]:
            shapes = [repeat_times] + list(self.send_tensor_shapes[0])
        else:
            shapes = self.send_tensor_shapes[0]
        rp_tensors = [
            torch.zeros(
                shapes,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_PREV']
        ]
        self.logs.append(f"[Rank{self.stage}] flush (sn {len(sn_tensors)}, sp {len(sp_tensors)}, rn {len(rn_tensors)}, rp {len(rp_tensors)}) @ {int(time.time() * 1000 - self.t_start):4} (ms)")
        if len(rn_tensors) > 0:
            if self.first_rn_ptr is None:
                self.first_rn_ptr = rn_tensors[0].data_ptr()
            self.logs.append(f"rn_tensors {[t.data_ptr() - self.first_rn_ptr for t in rn_tensors]}")
        if len(rp_tensors) > 0:
            if self.first_rp_ptr is None:
                self.first_rp_ptr = rp_tensors[0].data_ptr()
            self.logs.append(f"rp_tensors {[t.data_ptr() - self.first_rp_ptr for t in rp_tensors]}")
        # if self.stage == 1:
        #     print(f"[Rank{self.stage}] comm flush:", len(sn_tensors), len(sp_tensors), len(rn_tensors), len(rp_tensors))
        if get_args().profile:
            torch.cuda.nvtx.range_push(name)
        # if self.stage == 0:
        #     print(f"[Rank{self.stage}] C @ {int(time.time() * 1000 - self.first_f_start_time):4} (ms)", end="")
        # assert len(sn_tensors) + len(sp_tensors) + len(rn_tensors) + len(rp_tensors) == 1
        req, recv_time = fused_pipeline_ops(
            sp_tensors,
            rp_tensors,
            sn_tensors,
            rn_tensors,
            repeat_times,
            self.slow_links,
            self.stage,
        )
        # if self.stage == 0:
        #     print(f" to {int(time.time() * 1000 - self.first_f_start_time):4} (ms) rb {len(rn_tensors)}, rf {len(rp_tensors)}, sf {len(sn_tensors)}, sb {len(sp_tensors)}")
        # if len(rn_tensors) > 0:
        #     print(rn_tensors[0].sum())
        # if recv_time:
        #     if len(rp_tensors) + len(rn_tensors) == 2:
        #         self._update_recv_time_buffer(["prev", "next"], recv_time)
        #     elif len(rp_tensors) == 1:
        #         self._update_recv_time_buffer(["prev"], recv_time)
        #     elif len(rn_tensors) == 1:
        #         self._update_recv_time_buffer(["next"], recv_time)
        #     else:
        #         raise ValueError
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        # self.logs.append(f"len(ops) = {len(sn_tensors) + len(sp_tensors) + len(rn_tensors) + len(rp_tensors)}")
        # self.logs.append(f"reqs = {[id(r) for r in req]}")
        # We don't care about the reqs order here, all users need to all reqs to finish
        for x in self.communication_batch['RECV_NEXT']:
            self.buffer_map(x[0]).append(([rn_tensors.pop(0)], [req]))
        for x in self.communication_batch['RECV_PREV']:
            self.buffer_map(x[0]).append(([rp_tensors.pop(0)], [req]))
        log = None
        # if len(self.communication_batch['RECV_NEXT']) > 0:
        #     # log_string = f"[Rank{self.stage}] flush rb_buffer @ {int(time.time() * 1000 - self.t_start):4} (ms) {[t[0][0].sum().item() for t in self.recv_backward_buffer]}"
        #     log = f"[Rank{self.stage}] flush rb_buffer @ {int(time.time() * 1000 - self.t_start):4} (ms)"
        # if len(self.communication_batch['RECV_PREV']) > 0:
        #     # log_string = f"[Rank{self.stage}] flush rf_buffer @ {int(time.time() * 1000 - self.t_start):4} (ms) {[(id(t[0][0]), t[0][0].sum().item()) for t in self.recv_forward_buffer]}"
        #     log = f"[Rank{self.stage}] flush rf_buffer @ {int(time.time() * 1000 - self.t_start):4} (ms)"
        # if len(self.communication_batch['SEND_NEXT']) > 0:
        #     # log_string = f"[Rank{self.stage}] flush sn_tensor @ {int(time.time() * 1000 - self.t_start):4} (ms) {[t.sum().item() for t in sn_tensors]}"
        #     log = f"[Rank{self.stage}] flush sn_tensor @ {int(time.time() * 1000 - self.t_start):4} (ms)"
        # if len(self.communication_batch['SEND_PREV']) > 0:
        #     # log_string = f"[Rank{self.stage}] flush sp_tensor @ {int(time.time() * 1000 - self.t_start):4} (ms) {[t.sum().item() for t in sp_tensors]}"
        #     log = f"[Rank{self.stage}] flush sp_tensor @ {int(time.time() * 1000 - self.t_start):4} (ms)"
        self.send_handles.append([req])
        assert(not rn_tensors)
        assert(not rp_tensors)
        for direction in ['SEND_PREV', 'SEND_NEXT']:
            for idx, x in enumerate(self.communication_batch[direction]):
                if x[0].type == 'SEND_FORWARD':
                    deallocate_output_tensor(sp_tensors[idx] if direction == 'SEND_PREV' else sn_tensors[idx],
                                             self.config.deallocate_pipeline_outputs)
        # if log is not None:
        #     log += f" to {int(time.time() * 1000 - self.t_start):4} (ms)"
        #     self.logs.append(log)
        for k, v in self.communication_batch.items():
            v.clear()

    def add_communication(
        self,
        scheduled_node: auto_schedule.ScheduledNode,
        next_is_comm: bool,
        next_compute: auto_schedule.ScheduledNode
    ):
        if self.forward_only and 'BACKWARD' in scheduled_node.type:
            return
        self.communication_batch[self.direction_map(scheduled_node)].append(
            (scheduled_node, None))
        def is_consumer(scheduled_node, next_compute):
            if scheduled_node.minibatch == next_compute.minibatch:
                if scheduled_node.type == 'RECV_FORWARD' and next_compute.type == 'F':
                    return True
                if scheduled_node.type == 'RECV_BACKWARD' and next_compute.type == 'B':
                    return True
            return False
        # if (next_compute is not None and is_consumer(scheduled_node, next_compute)) or not next_is_comm or self.forward_only:
        # with torch.cuda.stream(self.comm_streams[0]):
        self.flush()

    def schedule_f(self, scheduled_node):
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = [None] * len(self.recv_tensor_shapes)
        else:
            # t_start = time.time()
            # t = Timer("receive-forward")
            # t.start()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record(stream=torch.cuda.current_stream())
            # start, end = start_a_timer()

            # for (_, next) in self.slow_links:
            #     if self.stage == next:
            #         time.sleep(self.sleep_time)
            #         break
            self.logs.append(f"[Rank{self.stage}] schedule_f @ {int(time.time() * 1000 - self.t_start):4} (ms)")
            input_tensor = self.recv_forward_buffer.pop(0)
            for h in input_tensor[1]:
                for hh in h:
                    # self.logs.append(f"before wait {id(hh)} @ {int(time.time() * 1000 - self.t_start):4} (ms)")
                    hh.wait()
            input_tensor = input_tensor[0]
            # self.logs.append(f"after wait {id(hh)} @ {int(time.time() * 1000 - self.t_start):4} (ms)")
            if input_tensor[0].dim() > len(self.send_tensor_shapes[0]):
                input_tensor = [input_tensor[0][0]]
            # self.logs.append(f"foobar wait {torch.count_nonzero(input_tensor[0]).item()} {id(hh)} @ {int(time.time() * 1000 - self.t_start):4} (ms)")
            self.logs.append(f"rf {input_tensor[0].data_ptr() - self.first_rp_ptr} {torch.count_nonzero(input_tensor[0]).item()} {input_tensor[0].sum().item()} @ {int(time.time() * 1000 - self.t_start):4} (ms)")
            self.logs.append(f"remain rf_buffer {[(t[0][0].data_ptr() - self.first_rp_ptr, torch.count_nonzero(t[0][0]).item(), t[0][0].sum().item()) for t in self.recv_forward_buffer]} @ {int(time.time() * 1000 - self.t_start):4} (ms)")
            # self.recv_time_buffer[self.new_iter_idx][1]["prev"][0] += time.time() - t_start
            # self.recv_time_buffer[self.new_iter_idx][1]["prev"][0] += t.elapsed()
            # end.record(stream=torch.cuda.current_stream())
            # end.synchronize()
            # self.recv_time_buffer[self.new_iter_idx][1]["prev"][0] += start.elapsed_time(end)
            # self.recv_time_buffer[self.new_iter_idx][1]["prev"][1] += 1
            # self._update_recv_time("prev", start, end)
        if get_args().profile:
            torch.cuda.nvtx.range_push(f'F{scheduled_node.minibatch}')
        # t_start = time.time()
        # t = Timer("forward")
        # t.start()
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record(stream=torch.cuda.current_stream())
        log_str = f"[Rank{self.stage}] F @ {int(time.time() * 1000 - self.t_start):4} (ms)"
        start, end = start_a_timer()
        if self.run_timer:
            ScheduleTimers.for_chunk(0).f_cnt += 1
            ScheduleTimers.for_chunk(0).f.start()
            mem_before = torch.cuda.memory_allocated()
        # with torch.cuda.stream(self.comp_stream):
        output_tensor = forward_step(
            self.forward_step_func,
            self.data_iterator,
            self.model,
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.config,
            self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
        )
        # print(f"RANK{torch.distributed.get_rank()}, FWD output tensor: {output_tensor}")
        if self.run_timer:
            ScheduleTimers.for_chunk(0).f.stop()
            ScheduleTimers.for_chunk(0).f_mem += torch.cuda.memory_allocated() - mem_before
        # self.exec_time_buffer[self.new_iter_idx][1]["F"][0] += time.time() - t_start
        # self.exec_time_buffer[self.new_iter_idx][1]["F"][0] += t.elapsed()
        # end.record(stream=torch.cuda.current_stream())
        # end.synchronize()
        # self.exec_time_buffer[self.new_iter_idx][1]["F"][0] += start.elapsed_time(end)
        # self.exec_time_buffer[self.new_iter_idx][1]["F"][1] += 1
        self._update_exec_time_buffer("F", start, end)
        etime = elapsed_time(start, end)
        self.logs.append(f"{log_str} to {int(time.time() * 1000 - self.t_start):4} (ms), cudaEventElapsed = {int(etime)} (ms)")
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        if not core.parallel_state.is_pipeline_last_stage():
            self.send_forward_buffer.append(output_tensor)
        if not self.forward_only:
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
            if core.parallel_state.is_pipeline_last_stage():
                deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)

    def schedule_b(self, scheduled_node):
        if not self.forward_only:
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = [None] * len(self.send_tensor_shapes)
            else:
                # t_start = time.time()
                # t = Timer("receive-backward")
                # t.start()
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record(stream=torch.cuda.current_stream())
                # start, end = start_a_timer()

                # for (prev, _) in self.slow_links:
                #     if self.stage == prev:
                #         time.sleep(self.sleep_time)
                #         break
                self.logs.append(f"[Rank{self.stage}] schedule_b @ {int(time.time() * 1000 - self.t_start):4} (ms)")
                output_tensor_grad = self.recv_backward_buffer.pop(0)
                for h in output_tensor_grad[1]:
                    for hh in h:
                        self.logs.append(f"wait {id(hh)}")
                        hh.wait()
                output_tensor_grad = output_tensor_grad[0]
                if output_tensor_grad[0].dim() > len(self.send_tensor_shapes[0]):
                    output_tensor_grad = [output_tensor_grad[0][0]]
                self.logs.append(f"rb {output_tensor_grad[0].data_ptr() - self.first_rn_ptr} {torch.count_nonzero(output_tensor_grad[0]).item()} {output_tensor_grad[0].sum().item()} @ {int(time.time() * 1000 - self.t_start):4} (ms)")
                self.logs.append(f"remain rb_buffer {[(t[0][0].data_ptr() - self.first_rn_ptr, torch.count_nonzero(t[0][0]).item(), t[0][0].sum().item()) for t in self.recv_backward_buffer]} @ {int(time.time() * 1000 - self.t_start):4} (ms)")
                # self.recv_time_buffer[self.new_iter_idx][1]["next"][0] += time.time() - t_start
                # self.recv_time_buffer[self.new_iter_idx][1]["next"][0] += t.elapsed()
                # end.record(stream=torch.cuda.current_stream())
                # end.synchronize()
                # self.recv_time_buffer[self.new_iter_idx][1]["next"][0] += start.elapsed_time(end)
                # self.recv_time_buffer[self.new_iter_idx][1]["next"][1] += 1
                # self._update_recv_time("next", start, end)
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'B{scheduled_node.minibatch}')
            # t_start = time.time()
            # t = Timer("backward-input")
            # t.start()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record(stream=torch.cuda.current_stream())
            log_str = f"[Rank{self.stage}] B @ {int(time.time() * 1000 - self.t_start):4} (ms)"
            start, end = start_a_timer()
            if self.run_timer:
                ScheduleTimers.for_chunk(0).b_cnt += 1
                ScheduleTimers.for_chunk(0).b.start()
                mem_before = torch.cuda.memory_allocated()
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, self.model_type,
                self.config
            )
            if self.run_timer:
                ScheduleTimers.for_chunk(0).b.stop()
                ScheduleTimers.for_chunk(0).b_mem += torch.cuda.memory_allocated() - mem_before
            # self.exec_time_buffer[self.new_iter_idx][1]["B"][0] += time.time() - t_start
            # self.exec_time_buffer[self.new_iter_idx][1]["B"][0] += t.elapsed()
            # end.record(stream=torch.cuda.current_stream())
            # end.synchronize()
            # self.exec_time_buffer[self.new_iter_idx][1]["B"][0] += start.elapsed_time(end)
            # self.exec_time_buffer[self.new_iter_idx][1]["B"][1] += 1
            self._update_exec_time_buffer("B", start, end)
            etime = elapsed_time(start, end)
            self.logs.append(f"{log_str} to {int(time.time() * 1000 - self.t_start):4} (ms), cudaEventElapsed = {int(etime)} (ms)")
            if get_args().profile:
                torch.cuda.nvtx.range_pop()
            self.send_backward_buffer.append(input_tensor_grad)
            WeightGradStore.flush()

    def schedule_w(self, scheduled_node, non_w_pending):
        if not self.forward_only and non_w_pending:
            # t_start = time.time()
            # t = Timer("backward-input")
            # t.start()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record(stream=torch.cuda.current_stream())
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'W{scheduled_node.minibatch}')
            log_str = f"[Rank{self.stage}] W @ {int(time.time() * 1000 - self.t_start):4} (ms)"
            start, end = start_a_timer()
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w_cnt += 1
                ScheduleTimers.for_chunk(0).w.start()
            WeightGradStore.pop()
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w.stop()
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][0] += time.time() - t_start
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][0] += t.elapsed()
            # end.record(stream=torch.cuda.current_stream())
            # end.synchronize()
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][0] += start.elapsed_time(end)
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][1] += 1
            self._update_exec_time_buffer("W", start, end)
            etime = elapsed_time(start, end)
            self.logs.append(f"{log_str} to {int(time.time() * 1000 - self.t_start):4} (ms), cudaEventElapsed = {int(etime)} (ms)")
            if get_args().profile:
                torch.cuda.nvtx.range_pop()

    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            self.no_sync_context = self.no_sync_func()
            self.no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None

    def prepare(
        self,
        schedule: List[auto_schedule.ScheduledNode],
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ):
        if isinstance(model, list):
            assert (
                len(model) == 1
            ), "non-interleaved pipeline parallelism does not support model chunking"
            model = model[0]
        if isinstance(data_iterator, list):
            assert (
                len(data_iterator) == 1
            ), "non-pipeline-parallel schedule does not support model chunking"
            data_iterator = data_iterator[0]

        config = get_model_config(model)
        if config.overlap_p2p_comm:
            raise ValueError(
                "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
            )
        # Disable async grad reductions
        no_sync_func = config.no_sync_func
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext
        self.no_sync_func = no_sync_func
        self.no_sync_context = None
        if not forward_only:
            ScheduleTimers.iter_counter += 1

        # Checkpoint the activations of partial Transformer layers in a number of micro-batches
        # within the maximum outstanding micro-batch backpropagations.
        # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
        # checkpoint partial Transformer layers (or skip checkpointing) and
        # the rest of micro-batches within a window of micro-batches checkpoint
        # all Transformer layers. The window of micro-batches is set by the maximum
        # outstanding backpropagations and becomes smaller at later pipeline stages.
        # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
        assert config.num_microbatches_with_partial_activation_checkpoints is None

        model_type = get_model_type(model)

        rank = parallel_state.get_pipeline_model_parallel_rank()
        recv_tensor_shapes = get_tensor_shapes(
            rank=rank - 1,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )
        send_tensor_shapes = get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )
        bootstrap_and_profile_p2p_communication(config, send_tensor_shapes,
                                                recv_tensor_shapes)

        run_timer = (
            get_args().zero_bubble_pipeline_timers_end_iter
            >= ScheduleTimers.iter_counter
            >= get_args().zero_bubble_pipeline_timers_start_iter
        )
        # run_timer = True

        self.config = config
        self.model_type = model_type
        self.recv_tensor_shapes = recv_tensor_shapes
        self.send_tensor_shapes = send_tensor_shapes
        self.run_timer = run_timer
        self.schedules = schedule
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.collect_non_loss_data = collect_non_loss_data
        self.forward_only = forward_only
        self._reset()
        self.it = 0

    def run_until_post_validation(self):
        optimizer = self.optimizer
        updated, grad_norm, rollback, succeed = None, None, None, None
        it = 0
        if optimizer.do_this_step:
            assert optimizer.do_prev_step
            if self.data_iterator is not None:
                self.data_iterator.clear_buffer()
                self.data_iterator.save_to_buffer()
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD"]:
                    next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                    next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                    next_compute = next_compute[0] if len(next_compute) > 0 else None
                    self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == 'F':
                    self.schedule_f(scheduled_node)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert succeed is not None
        else:
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD", "F"]:
                    if optimizer.do_prev_step and scheduled_node.type == "RECV_FORWARD":
                        next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                        next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                        next_compute = next_compute[0] if len(next_compute) > 0 else None
                        self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert not succeed
        if not succeed:
            if optimizer.do_prev_step:
                # send dummy recv_forward to clear send_forward request of last rank
                while it < len(self.schedules):
                    scheduled_node = self.schedules[it]
                    if scheduled_node.type == "RECV_FORWARD" and scheduled_node.rollback:
                        self.add_communication(scheduled_node, False, None)
                    it += 1
            self._reset()
            it = 0
        if succeed and self.data_iterator is not None:
            self.data_iterator.clear_buffer()
        if self.data_iterator is not None:
            self.data_iterator.pop_from_buffer()
        self.it = it
        return updated, grad_norm, rollback

    def plot(self):
        plt.figure(figsize=(10, 3 / self.num_stages + 1))
        ax = plt.subplot(111)
        maxt = 0
        for log in self.logs:
            terms = log.split()
            if terms[1][-1] not in ['F', 'B', 'W']:
                continue
            x, y = int(terms[3]), 0
            exec_time = int(terms[6]) - int(terms[3])
            rect = patches.Rectangle((x, y), exec_time, 1, linewidth=1, edgecolor='black', facecolor={'F': '#FCCCB3', 'W': '#FBE7A3', 'B': '#CBE4E4'}[terms[1][-1]])
            ax.add_patch(rect)
            ax.text(x + exec_time / 4, y + 1 / 2, terms[1])
            maxt = max(maxt, int(terms[6]))
        rect = patches.Rectangle((0, 0), maxt, 1, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
        ax.add_patch(rect)
        ax.set_title(f"Rank={self.stage}, S={self.num_stages}, B={self.num_microbatches}, Total Time = {maxt}")
        ax.set_xlim(0, ((maxt + 49) // 50) * 50)
        ax.set_ylim(0, 1)

    def run(self):
        #### Madoka: sync before each iter
        torch.cuda.synchronize()
        self.disable_grad_sync()
        # if self.iter_cnt % CHECK_INTERVAL == 0 and not self.forward_only:
        #     self._update_delay_info()
        self.iter_cnt += 1
        # if self.stage == 1 and self.num_microbatches == 6:
        #     self.schedules = [self.schedules[0], self.schedules[1], self.schedules[3], self.schedules[2], self.schedules[5], self.schedules[4], self.schedules[7], self.schedules[6], self.schedules[9], self.schedules[8], self.schedules[10], self.schedules[11]] + self.schedules[12:]
        #     # self.schedules = self.schedules[:3] + [self.schedules[4], self.schedules[3]] + self.schedules[5:]
        if self.forward_only or self.iter_cnt == 11:
            self.logs.append(f"[DP Rank{parallel_state.get_data_parallel_rank()}] [Rank{self.stage}] Schedule {' '.join([n.type for n in self.schedules if n.type in AUTO_SCHEDULE_COMMUNICATION_TYPES or n.type in ['F', 'B', 'W']])}")
            with open(f"./GPU{torch.distributed.get_rank()}_rank{self.stage}.log", 'w') as f:
                f.write('\n'.join(self.logs))
            # self.plot()
            # plt.savefig(f"/workspace/test-varuna/zerobubble/delay_s{self.stage}.png")
            exit(0)

        if get_args().profile:
            torch.cuda.nvtx.range_push(f'iter_{torch.distributed.get_rank()}_{ScheduleTimers.iter_counter}')

        it = self.it
        # if self.iter_cnt < SCHEDULE_UPDATE_START_ITER:
        if True:
            self.t_start = time.time() * 1000
            self.logs = []
            self.first_rn_ptr = None
            self.first_rp_ptr = None
            while it < len(self.schedules):
                # torch.cuda.synchronize()
                scheduled_node = self.schedules[it]
                self.logs.append(f"[Rank{self.stage}]=== schedule: {scheduled_node.type}, comm_batch: {self.communication_batch}, buffers: sf={len(self.send_forward_buffer)}, rf={len(self.recv_forward_buffer)}, sb={len(self.send_backward_buffer)}, rb={len(self.recv_backward_buffer)}")
                if "POST_VALIDATION" in scheduled_node.type:
                    pass
                elif scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                    next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                    next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                    next_compute = next_compute[0] if len(next_compute) > 0 else None
                    self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == 'F':
                    self.schedule_f(scheduled_node)
                elif scheduled_node.type == 'B':
                    self.schedule_b(scheduled_node)
                elif scheduled_node.type == 'W':
                    non_w_pending = any([node.type != 'W' for node in self.schedules[it + 1:]])
                    self.schedule_w(scheduled_node, non_w_pending)
                else:
                    raise ValueError(f"Unknown node type {scheduled_node.type}")
                it += 1
        else:
            while it < len(self.schedules[self.stage]):
                scheduled_node = self.schedules[self.stage][it]
                print(f"[Rank{self.stage} TODO] {scheduled_node.type}{scheduled_node.minibatch}")

                def send_and_recv(it: int, scheduled_node: MicroBatch):
                    """
                    Check if batched send and receive are necessary.
                    """
                    assert scheduled_node.type in ['F', 'B']
                    mb = scheduled_node.minibatch
                    diff_type = 'B' if scheduled_node.type == 'F' else 'F'
                    comm_stage = self.stage + 1 if scheduled_node.type == 'F' else self.stage - 1
                    for i in range(it + 1, len(self.schedules[self.stage])):
                        # Search for next f/b node
                        if self.schedules[self.stage][i].type == 'W':
                            continue
                        # Type of next f/b node is different from that of scheduled node
                        if self.schedules[self.stage][i].type == diff_type:
                            if scheduled_node.type == 'F':
                                f_mb = mb
                                b_mb = self.schedules[self.stage][i].minibatch
                            else:
                                f_mb = self.schedules[self.stage][i].minibatch
                                b_mb = mb
                            idx_b, idx_f = -1, -1
                            for j in range(len(self.schedules[comm_stage])):
                                node = self.schedules[comm_stage][j]
                                if node.type == 'B' and node.minibatch == b_mb:
                                    idx_b = j
                                if node.type == 'F' and node.minibatch == f_mb:
                                    idx_f = j
                                if idx_f != -1 and idx_b != -1:
                                    break
                            return idx_b < idx_f if scheduled_node.type == 'F' else idx_f < idx_b
                        # Type of next f/b node is the same as that of scheduled node
                        else:
                            # return False
                            if scheduled_node.type == 'F':
                                return False if self.stage == 0 else True
                            else:
                                return False if self.stage == self.num_stages - 1 else True
                    return False
                    
                if scheduled_node.type == "F":
                    if self.stage != 0:
                        self.add_communication(MicroBatch(scheduled_node.minibatch, "RECV_FORWARD"), False, scheduled_node)
                    self.schedule_f(scheduled_node)
                    if self.stage != self.num_stages - 1:
                        if it == len(self.schedules[self.stage]) - 1:
                            next_is_comm = False
                        else:
                            next_is_comm = send_and_recv(it, scheduled_node)
                        self.add_communication(MicroBatch(scheduled_node.minibatch, "SEND_FORWARD"), next_is_comm, None)
                elif scheduled_node.type == "B":
                    if self.stage != self.num_stages - 1:
                        self.add_communication(MicroBatch(scheduled_node.minibatch, "RECV_BACKWARD"), False, scheduled_node)
                    self.schedule_b(scheduled_node)
                    if self.stage != 0:
                        if it == len(self.schedules[self.stage]) - 1:
                            next_is_comm = False
                        else:
                            next_is_comm = send_and_recv(it, scheduled_node)
                        self.add_communication(MicroBatch(scheduled_node.minibatch, "SEND_BACKWARD"), next_is_comm, None)
                elif scheduled_node.type == "W":
                    non_w_pending = any([node.type != 'W' for node in self.schedules[self.stage][it + 1:]])
                    self.schedule_w(scheduled_node, non_w_pending)
                else:
                    raise ValueError(f"Unknown node type {scheduled_node.type}")
                it += 1
                # print(f"[Rank{self.stage} Done] {scheduled_node.type}{scheduled_node.minibatch}")
        self.it = it

        if get_args().profile:
            torch.cuda.nvtx.range_push('W')
        if not self.forward_only:
            pending_ws = WeightGradStore.weight_grad_queue[0].qsize()
            # t_start = time.time()
            # t = Timer("backward-weight")
            # t.start()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record(stream=torch.cuda.current_stream())
            log_str = f"[Rank{self.stage}] {pending_ws}W @ {int(time.time() * 1000 - self.t_start):4} (ms)"
            start, end = start_a_timer()
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w_cnt += pending_ws
                ScheduleTimers.for_chunk(0).w.start()
            WeightGradStore.clear(self.model)
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w.stop()
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][0] += time.time() - t_start
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][0] += t.elapsed()
            # end.record(stream=torch.cuda.current_stream())
            # end.synchronize()
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][0] += start.elapsed_time(end)
            # self.exec_time_buffer[self.new_iter_idx][1]["W"][1] += pending_ws
            self._update_exec_time_buffer("W", start, end, pending_ws)
            etime = elapsed_time(start, end)
            self.logs.append(f"{log_str} to {int(time.time() * 1000 - self.t_start):4} (ms), cudaEventElapsed = {int(etime)} (ms)")
        if get_args().profile:
            torch.cuda.nvtx.range_pop()  # W
            torch.cuda.nvtx.range_pop()  # iter

        for h in self.send_handles:
            for hh in h:
                for hhh in hh:
                    pass
                    hhh.wait()

        if not self.forward_only:
            # Launch any remaining grad reductions
            if self.no_sync_context is not None:
                self.enable_grad_sync()

            if self.config.finalize_model_grads_func is not None:
                # Finalize model grads (perform full grad all-reduce / reduce-scatter for
                # data parallelism, layernorm all-reduce for sequence parallelism).
                self.config.finalize_model_grads_func([self.model])

            if get_args().zero_bubble_pipeline_timers_end_iter == ScheduleTimers.iter_counter:
                ScheduleTimers.concluded = True
        if self.iter_cnt <= EXEC_TIME_PROFILE_END_ITER:
            self.exec_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][0] = self.iter_cnt
        self.recv_time_buffer[(self.iter_cnt - 1) % TIME_BUFFER_SIZE][0] = self.iter_cnt
        if self.iter_cnt == EXEC_TIME_PROFILE_END_ITER:
            self._update_exec_time()
            self._update_recv_time()
        # if not self.forward_only:
        #     # print("\n".join(self.logs))
        #     print(f"[Rank{self.stage}]", end="")
        #     print({task_type: int(t) for task_type, t in self.exec_time.items()}, end="")
        #     # if self.stage == 0:
        #     #     recv_froms = ["next"]
        #     # elif self.stage == self.num_stages - 1:
        #     #     recv_froms = ["prev"]
        #     # else:
        #     #     recv_froms = ["prev", "next"]
        #     # print({recv_from: [(self.recv_time_buffer[i][0], int(self.recv_time_buffer[i][1][recv_from][0] / self.recv_time_buffer[i][1][recv_from][1])) for i in range(TIME_BUFFER_SIZE) if len(self.recv_time_buffer[i]) > 0] for recv_from in recv_froms})
        #     # print({recv_from: int(self._get_recv_time(recv_from)) for recv_from in recv_froms})
        #     print({recv_from: int(t) for recv_from, t in self.recv_time.items()})

        #     # for i in range(TIME_BUFFER_SIZE):
        #     #     if len(self.exec_time_buffer[i]) > 0:
        #     #         print(f"[Iter{self.exec_time_buffer[i][0]:2}]",
        #     #         {k: (int(v[0] / v[1]), v[1]) for k, v in self.exec_time_buffer[i][1].items()})
        #     #         # print(ScheduleTimers.for_chunk(0).conclusion())
        #     #     else:
        #     #         print("[Empty]")
        #     # for i in range(TIME_BUFFER_SIZE):
        #     #     if len(self.recv_time_buffer[i]) > 0:
        #     #         print(f"[Iter{self.recv_time_buffer[i][0]:2}]",
        #     #         {k: (int(v[0] / v[1]), v[1]) for k, v in self.recv_time_buffer[i][1].items() if v[1] != 0})
        #     #         # print(ScheduleTimers.for_chunk(0).conclusion())
        #     #     else:
        #     #         print("[Empty]")

        # Prepare for the next iteration
        if self.iter_cnt + 1 <= EXEC_TIME_PROFILE_END_ITER:
            self.exec_time_buffer[self.iter_cnt % TIME_BUFFER_SIZE] = [None, {"F": [0, 0], "B": [0, 0], "W": [0, 0]}]
        self.recv_time_buffer[self.iter_cnt % TIME_BUFFER_SIZE] = [None, {"prev": [0, 0], "next": [0, 0]}]
        return self.forward_data_store

    def __call__(self, *args, **kwargs):
        if kwargs['forward_only']:
            self.prepare(*args, **kwargs)
            assert self.do_post_validation
            self.do_post_validation = True
            self.is_first_run = True
            return self.run()
        if not get_args().enable_optimizer_post_validation:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
            return self.run()
        # enable_optimizer_post_validation == True
        if self.is_first_run:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
            print('1' * 30)
        if self.do_post_validation:
            self.prepare(*args, **kwargs)
            result = self.run_until_post_validation()
            self.do_post_validation = False
            print('2' * 30)
        else:
            result = self.run()
            self.do_post_validation = True
            print('3' * 30)
        return result

zb_v_scheduler = None #ZeroBubbleVPipeScheduler()
zb_scheduler = ZeroBubbleScheduler()

def get_zb_scheduler_instance():
    if get_args().zero_bubble_v_schedule:
        global zb_v_scheduler
        return zb_v_scheduler
    else:
        global zb_scheduler
        return zb_scheduler


schedule = None
is_auto_schedule = False


def update_schedule(scheduler, f: List[int], b: List[int], w: List[int],
        c: int, f_mem: List[int], b_mem: List[int], w_mem: List[int],
        mem_limit: int):
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    ag_arguments = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(ag_arguments, (f,b,w,f_mem,b_mem,w_mem, mem_limit))
    assert len(ag_arguments) == torch.distributed.get_world_size()
    # Each value is an array of dimension (device, chunk)
    f,b,w,f_mem,b_mem,w_mem,mem_limit= zip(*ag_arguments)

    if is_second_last_pipeline_stage():
        print(f"rank {torch.distributed.get_rank()} Performing ILP with: f={f},\n b={b},\n w={w},\n c={c},\n f_mem={f_mem},\n b_mem={b_mem},\n w_mem={w_mem},\n mem_limit={mem_limit}")
        schedule = scheduler(
            pipeline_model_parallel_size,
            get_num_microbatches(),
            f, b, w,
            max(c, 1),
            f_mem, b_mem, w_mem,
            mem_limit,
        )
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, schedule)
        
    else:
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, None)
        schedule = list(filter(lambda x: x is not None, ag_result))
        assert len(schedule) == 1
        schedule = schedule[0]
    return schedule


def get_zero_bubble_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pipeline_model_parallel_size > 1, "zero bubble must be enabled with pipeline parallelism"

    args = get_args()
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    seq_length = args.seq_length
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx

    def wrapped_auto_schedule_forward_backward_func(func, scheduler):
        global schedule, is_auto_schedule
        if schedule is None:
            schedule = update_schedule(scheduler,
                f=[1000],
                b=[1000],
                w=[1000],
                c=1,
                f_mem=[f_mem_approx],
                b_mem=[0],
                w_mem=[-f_mem_approx],
                mem_limit=f_mem_approx * parallel_state.get_pipeline_model_parallel_world_size())
                # Using fixed 1p schedule
        if ScheduleTimers.concluded and not is_auto_schedule:
            conclusion = ScheduleTimers.joint_conclusion()
            # TODO(wanxy): Maybe an all-reduce here to collect global stats?
            print(f"rank {torch.distributed.get_rank()} profiling conclusion: {conclusion}")
            def estimate_free_memory_on_this_rank(old_schedule):
                (memory_free, memory_all) = [x // 1000000 for x in torch.cuda.mem_get_info()]
                memory_all = memory_all * get_args().zero_bubble_adaptive_memory_limit_percentile / 100
                activation_cost = 0
                stage = parallel_state.get_pipeline_model_parallel_rank()
                max_activation = 0
                for node in old_schedule[stage]:
                    chunk = node.chunk if hasattr(node, 'chunk') else 0
                    if node.type == 'F':
                        activation_cost += conclusion[4][chunk]
                    elif node.type == 'B':
                        activation_cost += conclusion[5][chunk]
                    elif node.type == 'W':
                        activation_cost += conclusion[6][chunk]
                    max_activation = max(activation_cost, max_activation)
                free_mem = memory_all - (torch.cuda.max_memory_allocated() // 1000000 - max_activation)

                print(f'estimated max free memory for activations on rank {torch.distributed.get_rank()} \
                    memory_free: {memory_free}, memory_all: {memory_all}, max_activation: {max_activation}, \
                    max_allocated: {torch.cuda.max_memory_allocated() // 1000000}, \
                    current_allocated: {torch.cuda.memory_allocated() // 1000000}, \
                    free_mem: {free_mem}')

                print(f'rank {torch.distributed.get_rank()} mem summary {torch.cuda.memory_summary()}')
                return free_mem
            schedule = update_schedule(scheduler,
                *conclusion,
                mem_limit=estimate_free_memory_on_this_rank(schedule))
            is_auto_schedule = True

        def wrap_schedule(**kwargs):
            return func(
                schedule=schedule[parallel_state.get_pipeline_model_parallel_rank()], **kwargs
            )
        return wrap_schedule

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem_limit):
            def avg_then_mid(a: List[List[float]]):
                a = [sum(x) / len(x) for x in a]
                return max(sorted(a)[len(a) // 2], 1)
            # For V schedule, we take average on each stage and then use mid value cross each stage.
            f_mid = avg_then_mid(f)
            b_mid = avg_then_mid(b)
            w_mid = avg_then_mid(w)
            if get_args().zero_bubble_v_schedule_mem_setup != 'zb':
                # Use fixed schedule for now
                ret = v_schedule_greedy.PipelineGraph(
                    nstages, nmb, get_args().zero_bubble_v_schedule_mem_setup, int(1000), int(1000), int(1000), int(1)
                ).get_schedule()
                return ret
            return v_schedule.PipelineGraph(
                nstages,
                nmb,
                f_mid,b_mid,w_mid,c,
                # V schedule does not consider memory differences between stages for now.
                f_mem=f_mem_approx, b_mem=b_mem_approx, w_mem=w_mem_approx,
                max_mem=None
                # Mem ignored for now
            ).get_v_schedule()
        if get_args().zero_bubble_v_schedule:
            global_zb_scheduler = get_zb_scheduler_instance()
            forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_scheduler, scheduler=scheduler)
            # forward_backward_func = wrapped_auto_schedule_forward_backward_func(forward_backward_pipelining_with_interleaving_auto_schedule,
            #                                                                     scheduler=scheduler)
        else:
            raise ValueError("got virtual pipeline parallel but v_schedule is disabled")
    else:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem_limit):
            f = [x[0] for x in f]
            b = [x[0] for x in b]
            w = [x[0] for x in w]
            # Using uniform f/b/w timing for now.
            f = [sorted(f)[len(f) // 2]] * len(f)
            b = [sorted(b)[len(b) // 2]] * len(b)
            w = [sorted(w)[len(w) // 2]] * len(w)
            f_mem = [x[0] for x in f_mem]
            b_mem = [x[0] for x in b_mem]
            w_mem = [x[0] for x in w_mem]

            if args.zero_bubble_max_pending_backward != 'auto':
                print(f'manual mem limit: {args.zero_bubble_max_pending_backward * max(f_mem[:2])}')
                mem_limit = [args.zero_bubble_max_pending_backward * max(f_mem[:2])] * len(f_mem)
            else:
                print(f'adaptive mem limit: {mem_limit}')
            
            return auto_schedule.auto_schedule(
                nstages,
                nmb,
                auto_schedule.GraphConfig(
                    cost_f=f,
                    cost_b=b,
                    cost_w=w,
                    cost_comm=c,
                    mem_f=f_mem,
                    mem_b=b_mem,
                    mem_w=w_mem,
                    max_mem=mem_limit,
                    print_scaling=1000
                ),
            )

        global_zb_scheduler = get_zb_scheduler_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_scheduler, scheduler=scheduler)

    return forward_backward_func
