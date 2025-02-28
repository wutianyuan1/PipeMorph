import contextlib
import os
import time
import socket
import redis
import torch
import torch.distributed
import torch.cuda.nvtx as nvtx
import memcpy
import numpy as np

from typing import Iterator, List, Union
from megatron import core, get_num_microbatches, print_rank_0
from megatron.core import parallel_state
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.pipeline_parallel.schedules import (
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)
from megatron.core.weight_grad_store import WeightGradStore
from pipeline_simulator.auto_schedule import ScheduledNode, auto_schedule, GraphConfig
from megatron.core.pipeline_parallel.event_timer import EventTimer
from megatron.core.pipeline_parallel.cpu_delegate import DelegateManager, addr_of, get_shm_signal, FLOAT16_NBYTES, UNREADY_SIGNAL, SEND_SIGNAL, RECV_SIGNAL_CPU, RECV_SIGNAL_GPU
from megatron.core.pipeline_parallel.profile_buffer import ProfileBuffer


AUTO_SCHEDULE_COMMUNICATION_TYPES = {'RECV_FORWARD', 'RECV_BACKWARD', 'SEND_FORWARD', 'SEND_BACKWARD'}
PREALLOC_BUFFER_SIZE = 32
SEND_WAY = 'shm'
RECV_WAY = 'shm'

log_file = None
timer_sync = True


def print_with_rank(message):
    global log_file
    print(f"[RANK{torch.distributed.get_rank()}] " + str(message), flush=True, file=log_file)


def get_ip_list():
    hostname = socket.gethostname()
    my_ip = torch.tensor([int(i) for i in socket.gethostbyname(hostname).split(".")], device='cuda', dtype=torch.int32)
    ip_tensor_list = [torch.zeros(4, dtype=torch.int32, device='cuda') for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(ip_tensor_list, my_ip)
    ip_list = [".".join([str(i) for i in ip_tensor.cpu().tolist()]) for ip_tensor in ip_tensor_list]
    print(f"My IP: {my_ip}, IP list: {ip_list}")
    return ip_list


class OurScheduler:
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
        self.no_sync_context = None
        self.no_sync_func = None
        self.optimizer = None
        self.forward_only = None
        self.is_first_run = True
        self.stage = parallel_state.get_pipeline_model_parallel_rank()
        self.num_stages = parallel_state.get_pipeline_model_parallel_world_size()
        self.iter_cnt = 0

        # Fail-slow injection
        self.slow_links = [(1, 2)]
        self.timer = EventTimer()
        self.comp_stream = torch.cuda.Stream(priority=-100)
        self.profile_buffer = None

    def _free_buffers(self):
        self.input_tensors = []
        self.output_tensors = []
        self.forward_data_store = []

    def _reset(self):
        self._free_buffers()

    def _should_delegate(self, scheduled_node: ScheduledNode, stage: int, peer_rank: int):
        return True

    def _is_delegated(self, task_id):
        return True

    def schedule_f(self, scheduled_node: ScheduledNode, op_id: int):
        delegate_id = op_id % self.delegate_manager.num_delegates
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = [None] * len(self.recv_tensor_shapes)
        else:
            if RECV_WAY == 'queue':
                input_tensor = [self.delegate_manager.queues[f'recv_forward_task_{delegate_id}'].get().cuda().requires_grad_()]
            else:
                shm = self.delegate_manager.shms[f'recv_forward_shm_{delegate_id}']
                idx = op_id // self.delegate_manager.num_delegates
                while True:
                    signal = get_shm_signal(shm.buf, idx)
                    if signal.item() == RECV_SIGNAL_CPU:
                        signal[0] = RECV_SIGNAL_GPU
                        break
                cpu_tensor_addr = addr_of(shm.buf, self.num_microbatches * FLOAT16_NBYTES + idx * self.data_size)
                gpu_tensor = torch.zeros(self.send_tensor_shapes[0], requires_grad=True, device=torch.cuda.current_device(), dtype=self.config.pipeline_dtype)
                memcpy.cudaH2DAsync(cpu_tensor_addr, gpu_tensor, self.data_size)
                cpu_signal_addr = addr_of(shm.buf, idx * FLOAT16_NBYTES)
                memcpy.cudaH2DAsync(cpu_signal_addr, self.rf_signals[op_id], FLOAT16_NBYTES)
                cpu_signal = get_shm_signal(shm.buf, idx)
                while True:
                    if self.rf_signals[op_id].item() == RECV_SIGNAL_GPU:
                        self.rf_signals[op_id][0] = UNREADY_SIGNAL
                        cpu_signal[0] = UNREADY_SIGNAL
                        break
                input_tensor = [gpu_tensor]
            assert torch.sum(input_tensor[0]) != 111.111
        with torch.cuda.stream(self.comp_stream):
            timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
            with nvtx.range(scheduled_node.type):
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
            self.timer.end(timer_id, "F")
        if not core.parallel_state.is_pipeline_last_stage():
            if SEND_WAY == 'queue':
                self.delegate_manager.queues[f'send_forward_task_{delegate_id}'].put(output_tensor[0].detach().cpu())
            else:
                shm = self.delegate_manager.shms[f'send_forward_shm_{delegate_id}']
                idx = op_id // self.delegate_manager.num_delegates
                cpu_tensor_addr = addr_of(shm.buf, self.num_microbatches * FLOAT16_NBYTES + idx * self.data_size)
                memcpy.cudaD2HAsync(output_tensor[0].detach(), cpu_tensor_addr, self.data_size)
                cpu_signal_addr = addr_of(shm.buf, idx * FLOAT16_NBYTES)
                signal = torch.tensor((SEND_SIGNAL), dtype=torch.float16, device=torch.cuda.current_device())
                memcpy.cudaD2HAsync(signal, cpu_signal_addr, FLOAT16_NBYTES)
        if not self.forward_only:
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
        deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)

    def schedule_b(self, scheduled_node: ScheduledNode, op_id: int):
        if self.forward_only:
            return
        delegate_id = op_id % self.delegate_manager.num_delegates
        if not self.forward_only:
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = [None] * len(self.send_tensor_shapes)
            else:
                if RECV_WAY == 'queue':
                    x = self.delegate_manager.queues[f'recv_backward_task_{delegate_id}'].get()
                    output_tensor_grad = [x.cuda().requires_grad_()]
                else:
                    shm = self.delegate_manager.shms[f'recv_backward_shm_{delegate_id}']
                    idx = op_id // self.delegate_manager.num_delegates
                    while True:
                        signal = get_shm_signal(shm.buf, idx)
                        if signal.item() == RECV_SIGNAL_CPU:
                            signal[0] = RECV_SIGNAL_GPU
                            break
                    cpu_tensor_addr = addr_of(shm.buf, self.num_microbatches * FLOAT16_NBYTES + idx * self.data_size)
                    gpu_tensor = torch.zeros(self.send_tensor_shapes[0], requires_grad=True, device=torch.cuda.current_device(), dtype=self.config.pipeline_dtype)
                    memcpy.cudaH2DAsync(cpu_tensor_addr, gpu_tensor, self.data_size)
                    cpu_signal_addr = addr_of(shm.buf, idx * FLOAT16_NBYTES)
                    memcpy.cudaH2DAsync(cpu_signal_addr, self.rb_signals[op_id], FLOAT16_NBYTES)
                    cpu_signal = get_shm_signal(shm.buf, idx)
                    while True:
                        if self.rb_signals[op_id].item() == RECV_SIGNAL_GPU:
                            self.rb_signals[op_id][0] = UNREADY_SIGNAL
                            cpu_signal[0] = UNREADY_SIGNAL
                            break
                    output_tensor_grad = [gpu_tensor]
                assert torch.sum(output_tensor_grad[0]) != 111.111

            with torch.cuda.stream(self.comp_stream):
                timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
                with nvtx.range(scheduled_node.type):
                    input_tensor_grad = backward_step(
                        input_tensor, output_tensor, output_tensor_grad, self.model_type,
                        self.config
                    )
                self.timer.end(timer_id, "B")
            if not parallel_state.is_pipeline_first_stage():
                if SEND_WAY == 'queue':
                    self.delegate_manager.queues[f'send_backward_task_{delegate_id}'].put(input_tensor_grad[0].detach().cpu())
                else:    
                    shm = self.delegate_manager.shms[f'send_backward_shm_{delegate_id}']
                    idx = op_id // self.delegate_manager.num_delegates
                    cpu_tensor_addr = addr_of(shm.buf, self.num_microbatches * FLOAT16_NBYTES + idx * self.data_size)
                    memcpy.cudaD2HAsync(input_tensor_grad[0].detach(), cpu_tensor_addr, self.data_size)
                    cpu_signal_addr = addr_of(shm.buf, idx * FLOAT16_NBYTES)
                    signal = torch.tensor((SEND_SIGNAL), dtype=torch.float16, device=torch.cuda.current_device())
                    memcpy.cudaD2HAsync(signal, cpu_signal_addr, FLOAT16_NBYTES)
            WeightGradStore.flush()

    def schedule_w(self, scheduled_node: ScheduledNode, non_w_pending: bool):
        with torch.cuda.stream(self.comp_stream):
            if not self.forward_only and non_w_pending:
                timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
                WeightGradStore.pop()
                self.timer.end(timer_id, "W")

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
        schedule: List[ScheduledNode],
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        profile_buffer: ProfileBuffer = None
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
        self.config = config
        self.model_type = model_type
        self.recv_tensor_shapes = recv_tensor_shapes
        self.send_tensor_shapes = send_tensor_shapes
        self.schedules = schedule
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.collect_non_loss_data = collect_non_loss_data
        self.forward_only = forward_only
        if profile_buffer:
            self.profile_buffer = profile_buffer
        self._reset()


        ip_list = get_ip_list()
        tp_dp = parallel_state.get_tensor_model_parallel_world_size() * parallel_state.get_data_parallel_world_size()
        my_rank = torch.distributed.get_rank()
        my_ip = ip_list[my_rank]
        prev_ip = ip_list[my_rank - tp_dp] if self.stage != 0 else None
        next_ip = ip_list[my_rank + tp_dp] if self.stage != self.num_stages - 1 else None
        self.delegate_manager = DelegateManager(self.stage, self.num_stages, my_ip, prev_ip, next_ip, send_tensor_shapes[0], self.config.pipeline_dtype, num_delegates=3, ipc_way={'send_way': SEND_WAY, 'recv_way': RECV_WAY}, num_microbatches=num_microbatches)
        assert send_tensor_shapes[0] == recv_tensor_shapes[0]
        self.data_size = torch.prod(torch.tensor(send_tensor_shapes[0])).item() * FLOAT16_NBYTES
        if RECV_WAY == "shm":
            if not core.parallel_state.is_pipeline_first_stage():
                self.rf_signals = [torch.zeros((1,), device=torch.cuda.current_device(), dtype=self.config.pipeline_dtype) for _ in range(self.num_microbatches)]
            if not core.parallel_state.is_pipeline_last_stage():
                self.rb_signals = [torch.zeros((1,), device=torch.cuda.current_device(), dtype=self.config.pipeline_dtype) for _ in range(self.num_microbatches)]

    def run(self):
        # Add a sync here
        torch.distributed.barrier()
        torch.cuda.synchronize()
        # Actual training loigc starts here!
        self.disable_grad_sync()
        self.iter_cnt += 1
        global log_file
        if log_file is None:
            log_file = open(f"./GPU{torch.distributed.get_rank()}_rank{torch.distributed.get_rank()}_CPU.log", 'w')

        # Get a unified timestamp across all ranks
        ts = torch.tensor([time.time() * 1000], dtype=torch.float64, device=f'cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}')
        torch.distributed.all_reduce(ts, op=torch.distributed.ReduceOp.MAX)
        self.t_start = ts.item()

        fs = [i.type for i in self.schedules if i.type == 'F']
        self.delegate_manager.start_iter(len(fs))

        # Run this schedule.
        it = 0
        fid, bid = 0, 0
        max_cuda_reserved_mem = 0
        while it < len(self.schedules):
            scheduled_node = self.schedules[it]
            it += 1
            if 'POST' in scheduled_node.type:
                continue
            if scheduled_node.type == 'F':
                self.schedule_f(scheduled_node, fid)
                fid += 1
            elif scheduled_node.type == 'B':
                self.schedule_b(scheduled_node, bid)
                bid += 1
            elif scheduled_node.type == 'W':
                with nvtx.range(scheduled_node.type):
                    non_w_pending = any([node.type != 'W' for node in self.schedules[it:]])
                    self.schedule_w(scheduled_node, non_w_pending)
            mem = torch.cuda.memory_reserved(torch.cuda.current_device())
            max_cuda_reserved_mem = max(max_cuda_reserved_mem, mem)
        # print(f"[Node{self.stage}] Max reserved memory: {max_cuda_reserved_mem / (1024 ** 3)} GB")

        # Finalize, process the pending Ws
        pending_ws = WeightGradStore.weight_grad_queue[0].qsize()
        with torch.cuda.stream(self.comp_stream):
            with nvtx.range(f"{pending_ws}W"):
                timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
                WeightGradStore.clear(self.model)
                self.timer.end(timer_id, f"{pending_ws}W")

        print_with_rank(f"Iteration {self.iter_cnt}")
        self.timer.print_all(print_func=print_with_rank)

        if not self.forward_only:
            # Launch any remaining grad reductions
            if self.no_sync_context is not None:
                self.enable_grad_sync()

            if self.config.finalize_model_grads_func is not None:
                # Finalize model grads (perform full grad all-reduce / reduce-scatter for
                # data parallelism, layernorm all-reduce for sequence parallelism).
                self.config.finalize_model_grads_func([self.model])
        return self.forward_data_store

    def __call__(self, *args, **kwargs):
        # If we receive this signal, kill all CPU delegates.
        if 'terminate_signal' in kwargs and kwargs['terminate_signal']:
            self.delegate_manager.terminate()
            return
        # HACK: just skip the evaluation iters
        if kwargs['forward_only']:
            return self.forward_data_store
        if self.is_first_run:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
        return self.run()


our_scheduler = OurScheduler()
schedule = None
redis_client = None
delay_links, delay_time = [], 0.0
reschedule_at_next_iter = False


def get_our_scheduler_instance():
    global our_scheduler
    return our_scheduler


def update_schedule(num_stages, num_microbatches, profile_exec_times):
    global schedule, redis_client, delay_links, delay_time, reschedule_at_next_iter
    if redis_client is None:
        redis_client = redis.StrictRedis(host=os.environ['MASTER_ADDR'], port=int(os.environ.get('REDIS_PORT', '6379')))

    # If the previous iteration requires a re-schedule, clear the old schedule
    if reschedule_at_next_iter:
        reschedule_at_next_iter = False
        schedule = None

    delay_links_reply = redis_client.get("slow_links")
    new_delay_links = []
    new_delay_time = 0.0
    if delay_links_reply is not None:
        delay_links_str = delay_links_reply.decode()
        if delay_links_str != "":
            for pair in delay_links_str.split(","):
                start, end = pair.split("_")
                new_delay_links.append((int(start), int(end)))
    delay_time_reply = redis_client.get("sleep_time")
    if delay_time_reply is not None:
        new_delay_time = float(delay_time_reply)
    # Check if the delay info has been changed
    if new_delay_time != delay_time or new_delay_links != delay_links:
        delay_links = new_delay_links
        delay_time = new_delay_time
        reschedule_at_next_iter = True  # Require re-schedule

    if schedule is None:
        is_valid_profile = True if (profile_exec_times is not None and profile_exec_times[0, 0] != 0.0) else False
        schedule = auto_schedule(
            num_stages,
            num_microbatches,
            GraphConfig(
                mem_f=[1000], mem_b=[-500], mem_w=[-500],
                cost_f=np.round(profile_exec_times[0]).astype(int).tolist() if is_valid_profile else [10] * num_stages,
                cost_b=np.round(profile_exec_times[1]).astype(int).tolist() if is_valid_profile else [10] * num_stages,
                cost_w=np.round(profile_exec_times[2]).astype(int).tolist() if is_valid_profile else [10] * num_stages,
                cost_comm=0
            ),
            delay_links,
            delay_time * 1000.0  # convert s to ms
        )
        for stage in range(num_stages):
            print_rank_0(f'Stage {stage}')
            print_rank_0([node.type for node in schedule[stage]])
    return schedule


def get_zero_bubble_cpu_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pipeline_model_parallel_size > 1, "[OurSchedule] must enable pipeline parallelism"
    scheduler_instance = get_our_scheduler_instance()
    if scheduler_instance.profile_buffer is not None:
        profile_exec_times = scheduler_instance.profile_buffer.get_stage_exec_times()
    else:
        profile_exec_times = None
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    global_schedule = update_schedule(pipeline_model_parallel_size, get_num_microbatches(), profile_exec_times)
    scheduler_instance.schedules = global_schedule[pp_rank]

    def forward_backward_func(**kwargs):
        return scheduler_instance(schedule=global_schedule[pp_rank], **kwargs)

    return forward_backward_func
