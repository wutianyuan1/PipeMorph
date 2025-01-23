import contextlib
import os
import time
import uuid
import torch
import torch.distributed
import torch.cuda.nvtx as nvtx

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
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.weight_grad_store import WeightGradStore
from pipeline_simulator.auto_schedule import ScheduledNode, auto_schedule, GraphConfig
from megatron.core.pipeline_parallel.event_timer import EventTimer
from megatron.core.pipeline_parallel.delegate import DelegateManager


AUTO_SCHEDULE_COMMUNICATION_TYPES = {'RECV_FORWARD', 'RECV_BACKWARD', 'SEND_FORWARD', 'SEND_BACKWARD'}
PREALLOC_BUFFER_SIZE = 32

log_file = None
timer_sync = True


def print_with_rank(message):
    global log_file
    print(f"[RANK{torch.distributed.get_rank()}] " + str(message), flush=True, file=log_file)


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
        self.repeat_times = 3
        self.timer = EventTimer()

        # Delegation
        self.taskid = 0
        self.dele_ids = set()
        self.delegate_manager = DelegateManager(
            {'delegate_rank': self.stage, 'delegate_world_size': self.num_stages}, ndeles=2)

    def _free_buffers(self):
        self.input_tensors = []
        self.output_tensors = []
        self.send_forward_buffer = []
        self.recv_forward_id_buffer = []
        self.send_backward_buffer = []
        self.recv_backward_id_buffer = []
        self.forward_data_store = []

    def _reset(self):
        self._free_buffers()
        self.dele_ids = set()

    def _should_delegate(self, scheduled_node: ScheduledNode, stage: int, peer_rank: int):
        return True

    def _is_delegated(self, task_id):
        if task_id in self.dele_ids:
            return True
        return False

    def get_buffer(self, scheduled_node: ScheduledNode):
        def get_free_pool_id(free_id_list):
            # Current policy: pop the first item
            if len(free_id_list) == 0:
                print_with_rank("No free pool slots!")
                raise RuntimeError("No free pool slots!")
            return free_id_list.pop(0)

        if scheduled_node.type == 'RECV_FORWARD':
            pool_id = get_free_pool_id(self.free_rp_list)
            return self.rp_tensor_pool[pool_id], pool_id
        elif scheduled_node.type == 'RECV_BACKWARD':
            pool_id = get_free_pool_id(self.free_rn_list)
            return self.rn_tensor_pool[pool_id], pool_id
        elif scheduled_node.type == 'SEND_FORWARD':
            return self.send_forward_buffer.pop(0)[0], -1
        elif scheduled_node.type == 'SEND_BACKWARD':
            return self.send_backward_buffer.pop(0)[0], -1
        else:
            print_with_rank(f"Unsupported scheduled node: {scheduled_node}")

    def add_communication(self, scheduled_node: ScheduledNode, rest_nodes: List[ScheduledNode]):
        ops_map = {
            'RECV_FORWARD': (torch.distributed.irecv, get_pipeline_model_parallel_prev_rank()),
            'RECV_BACKWARD': (torch.distributed.irecv, get_pipeline_model_parallel_next_rank()),
            'SEND_FORWARD': (torch.distributed.isend, get_pipeline_model_parallel_next_rank()),
            'SEND_BACKWARD': (torch.distributed.isend, get_pipeline_model_parallel_prev_rank())
        }
        comm_func, peer_rank = ops_map[scheduled_node.type]
        buffer, pool_id = self.get_buffer(scheduled_node)

        task_id = str(uuid.uuid4())  # self.taskid
        self.taskid += 1
        if scheduled_node.type == 'RECV_FORWARD':
            self.recv_forward_id_buffer.append((pool_id, task_id))
        elif scheduled_node.type == 'RECV_BACKWARD':
            self.recv_backward_id_buffer.append((pool_id, task_id))

        if self._should_delegate(scheduled_node, self.stage, peer_rank):
            timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
            if 'FORWARD' in scheduled_node.type:
                self.delegate_manager.schedule_f(
                    {
                        'type': 'recv' if scheduled_node.type.startswith('RECV') else 'send',
                        'buffer': buffer.detach(),
                        'peer_rank': peer_rank,
                        'task_id': task_id
                    }
                )
            else:
                self.delegate_manager.schedule_b(
                    {
                        'type': 'recv' if scheduled_node.type.startswith('RECV') else 'send',
                        'buffer': buffer.detach(),
                        'peer_rank': peer_rank,
                        'task_id': task_id
                    }
                )
            self.dele_ids.add(task_id)
            self.timer.end(timer_id, scheduled_node.type)
        else:
            timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
            future_handle = comm_func(buffer, peer_rank, get_pipeline_model_parallel_group())
            self.timer.end(timer_id, scheduled_node.type)

            # Add future handles to the signal list, we should wait for it when using corresponding buffers
            if scheduled_node.type == 'RECV_FORWARD':
                self.rp_pool_signals[pool_id] = future_handle
            elif scheduled_node.type == 'RECV_BACKWARD':
                self.rn_pool_signals[pool_id] = future_handle

        # De-allocate the input tensor buffers right after it is sent to the next stage
        if scheduled_node.type == 'SEND_FORWARD':
            deallocate_output_tensor(buffer, self.config.deallocate_pipeline_outputs)

    def schedule_f(self, scheduled_node: ScheduledNode):
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = [None] * len(self.recv_tensor_shapes)
        else:
            # Madoka: get the tensor by id from pool when we use it, push the pool_id back after use
            pool_id, task_id = self.recv_forward_id_buffer.pop(0)
            # Wait for the corresponding recv task to complete
            if self._is_delegated(task_id):
                self.delegate_manager.wait_f(task_id)
            else:
                # Wait for data transfer ready
                assert self.rp_pool_signals[pool_id] is not None
                self.rp_pool_signals[pool_id].wait()
            input_tensor = [self.rp_tensor_pool[pool_id]]
        timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
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
            self.send_forward_buffer.append(output_tensor)
        if not self.forward_only:
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
            if core.parallel_state.is_pipeline_last_stage():
                deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)
        # Madoka: push back the pool_id we used, and cleanup the future signal
        if not core.parallel_state.is_pipeline_first_stage():
            self.free_rp_list.append(pool_id)
            self.rp_pool_signals[pool_id] = None

    def schedule_b(self, scheduled_node: ScheduledNode):
        if not self.forward_only:
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = [None] * len(self.send_tensor_shapes)
            else:
                # Madoka: get the tensor by id from pool when we use it, push the pool_id back after use
                pool_id, task_id = self.recv_backward_id_buffer.pop(0)
                if self._is_delegated(task_id):
                    # Wait for the corresponding recv task to complete
                    self.delegate_manager.wait_b(task_id)
                else:
                    # Wait for data transfer ready
                    assert self.rn_pool_signals[pool_id] is not None
                    self.rn_pool_signals[pool_id].wait()
                output_tensor_grad = [self.rn_tensor_pool[pool_id]]

            timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, self.model_type,
                self.config
            )
            self.timer.end(timer_id, "B")
            self.send_backward_buffer.append(input_tensor_grad)
            WeightGradStore.flush()
            # Madoka: push back the pool_id we used, and cleanup the future signal
            if not core.parallel_state.is_pipeline_last_stage():
                self.free_rn_list.append(pool_id)
                self.rn_pool_signals[pool_id] = None

    def schedule_w(self, scheduled_node: ScheduledNode, non_w_pending: bool):
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

    def _pre_alloc_buffers(self):
        # Stacked tensor
        if self.stage in [p for (p, n) in self.slow_links]:
            shapes = [self.repeat_times] + list(self.send_tensor_shapes[0])
        else:
            shapes = self.send_tensor_shapes[0]
        rn_tensors = [
            torch.zeros(
                self.send_tensor_shapes[0],
                # shapes,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for _ in range(PREALLOC_BUFFER_SIZE)
        ]
        # Stacked tensor
        if self.stage in [n for (p, n) in self.slow_links]:
            shapes = [self.repeat_times] + list(self.send_tensor_shapes[0])
        else:
            shapes = self.send_tensor_shapes[0]
        rp_tensors = [
            torch.zeros(
                self.send_tensor_shapes[0],
                # shapes,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for _ in range(PREALLOC_BUFFER_SIZE)
        ]
        self.rn_tensor_pool = rn_tensors
        self.rp_tensor_pool = rp_tensors
        self.rn_pool_signals = [None for _ in range(PREALLOC_BUFFER_SIZE)]
        self.rp_pool_signals = [None for _ in range(PREALLOC_BUFFER_SIZE)]
        self.free_rn_list = list(range(PREALLOC_BUFFER_SIZE))
        self.free_rp_list = list(range(PREALLOC_BUFFER_SIZE))

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
        self._reset()
        self._pre_alloc_buffers()

    def run(self):
        # Add a sync here
        torch.distributed.barrier()
        torch.cuda.synchronize()
        # Actual training loigc starts here!
        self.disable_grad_sync()
        self.iter_cnt += 1
        # Now, only run MAXITERS iterations for testing and record the last iter's performance
        MAXITERS = 6
        global log_file
        if self.iter_cnt == MAXITERS:
            self.delegate_manager.terminate()
            exit(0)
        log_file = open(f"/workspace/test-varuna/zerobubble/GPU{torch.distributed.get_rank()}_rank{torch.distributed.get_rank()}.log", 'w') if self.iter_cnt == MAXITERS - 1 else open(os.devnull, 'w')

        # Get a unified timestamp across all ranks
        ts = torch.tensor([time.time() * 1000], dtype=torch.float64, device=f'cuda:{torch.distributed.get_rank()}')
        torch.distributed.all_reduce(ts, op=torch.distributed.ReduceOp.MAX)
        self.t_start = ts.item()

        # Run this schedule.
        it = 0
        while it < len(self.schedules):
            scheduled_node = self.schedules[it]
            it += 1
            if 'POST' in scheduled_node.type:
                continue
            if scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                with nvtx.range(scheduled_node.type):
                    self.add_communication(scheduled_node, self.schedules[it:])
            elif scheduled_node.type == 'F':
                with nvtx.range(scheduled_node.type):
                    self.schedule_f(scheduled_node)
            elif scheduled_node.type == 'B':
                with nvtx.range(scheduled_node.type):
                    self.schedule_b(scheduled_node)
            elif scheduled_node.type == 'W':
                with nvtx.range(scheduled_node.type):
                    non_w_pending = any([node.type != 'W' for node in self.schedules[it:]])
                    self.schedule_w(scheduled_node, non_w_pending)

        # Finalize, process the pending Ws
        pending_ws = WeightGradStore.weight_grad_queue[0].qsize()
        with nvtx.range(f"{pending_ws}W"):
            timer_id = self.timer.start(self.t_start, eager_sync=timer_sync)
            WeightGradStore.clear(self.model)
            self.timer.end(timer_id, f"{pending_ws}W")

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
        if self.is_first_run:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
        return self.run()


our_scheduler = OurScheduler()
schedule = None


def get_our_scheduler_instance():
    global our_scheduler
    return our_scheduler


def update_schedule(num_stages, num_microbatches):
    global schedule
    if schedule is None:
        schedule = auto_schedule(
            num_stages,
            num_microbatches,
            GraphConfig(
                mem_f=[1000], mem_b=[-500], mem_w=[-500],
                cost_f=[1000] * 4, cost_b=[1000] * 4, cost_w=[1000] * 4, cost_comm=0
            )
        )
        for stage in range(num_stages):
            print_rank_0(f'Stage {stage}')
            print_rank_0([node.type for node in schedule[stage]])
    return schedule


def get_zero_bubble_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pipeline_model_parallel_size > 1, "[OurSchedule] must enable pipeline parallelism"
    scheduler_instance = get_our_scheduler_instance()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    global_schedule = update_schedule(pipeline_model_parallel_size, get_num_microbatches())
    forward_backward_func = lambda **kwargs: scheduler_instance(schedule=global_schedule[pp_rank], **kwargs)
    return forward_backward_func
