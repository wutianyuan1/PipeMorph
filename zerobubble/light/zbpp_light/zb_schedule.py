import contextlib
import itertools
from typing import Iterator, List, Union

import torch

from megatron import core, get_args, get_num_microbatches, print_rank_0
from megatron.core import parallel_state
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
from zbpp_light import auto_schedule
from zbpp_light.weight_grad_store import WeightGradStore


AUTO_SCHEDULE_COMMUNICATION_TYPES = {'RECV_FORWARD', 'RECV_BACKWARD', 'SEND_FORWARD', 'SEND_BACKWARD'}


def fused_pipeline_ops(
    tensor_send_prev: List[torch.Tensor],
    tensor_recv_prev: List[torch.Tensor],
    tensor_send_next: List[torch.Tensor],
    tensor_recv_next: List[torch.Tensor],
):
    ops = []
    group = get_pipeline_model_parallel_group()
    for t in tensor_send_prev:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    for t in tensor_recv_prev:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    for t in tensor_send_next:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    for t in tensor_recv_next:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs

class ZeroBubbleScheduler:

    def __init__(self):
        self._reset()

        self.schedules = None
        self.send_tensor_shapes = None
        self.recv_tensor_shapes = None
        self.config = None
        self.forward_step_func = None
        self.data_iterator = None
        self.model = None
        self.model_type = None
        self.num_microbatches = None
        self.collect_non_loss_data = None
        self.forward_only = None
        self.no_sync_context = None
        self.no_sync_func = None

        self.do_post_validation = False
        self.is_first_run = True
        self.optimizer = None

    def _free_buffers(self):
        self.input_tensors = []
        self.output_tensors = []
        self.send_forward_buffer = []
        self.recv_forward_buffer = []
        self.send_backward_buffer = []
        self.recv_backward_buffer = []
        self.forward_data_store = []

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

    def get_schedules(self):
        if self.schedules is None:
            bootstrap_p2p_communication(self.config)
            self.schedules = auto_schedule.auto_schedule(
                parallel_state.get_pipeline_model_parallel_world_size(),
                get_num_microbatches())[parallel_state.get_pipeline_model_parallel_rank()]

        return self.schedules

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

    def flush(self):
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

        rn_tensors = [
            torch.empty(
                self.send_tensor_shapes[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_NEXT']
        ]
        rp_tensors = [
            torch.empty(
                self.send_tensor_shapes[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_PREV']
        ]
        req = fused_pipeline_ops(
            sp_tensors,
            rp_tensors,
            sn_tensors,
            rn_tensors
        )
        # We don't care about the reqs order here, all users need to all reqs to finish
        for x in self.communication_batch['RECV_NEXT']:
            self.buffer_map(x[0]).append(([rn_tensors.pop(0)], [req]))
        for x in self.communication_batch['RECV_PREV']:
            self.buffer_map(x[0]).append(([rp_tensors.pop(0)], [req]))
        self.send_handles.append([req])
        assert(not rn_tensors)
        assert(not rp_tensors)
        for direction in ['SEND_PREV', 'SEND_NEXT']:
            for id, x in enumerate(self.communication_batch[direction]):
                if x[0].type == 'SEND_FORWARD':
                    deallocate_output_tensor(sp_tensors[id] if direction == 'SEND_PREV' else sn_tensors[id],
                                             self.config.deallocate_pipeline_outputs)
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
        if (next_compute is not None and is_consumer(scheduled_node, next_compute)) or not next_is_comm or self.forward_only:
            self.flush()

    def schedule_f(self, scheduled_node):
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = [None] * len(self.recv_tensor_shapes)
        else:
            input_tensor = self.recv_forward_buffer.pop(0)
            for h in input_tensor[1]:
                for hh in h:
                    hh.wait()
            input_tensor = input_tensor[0]
        
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

        if not core.parallel_state.is_pipeline_last_stage():
            self.send_forward_buffer.append(output_tensor)
        if not self.forward_only:
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
            if core.parallel_state.is_pipeline_last_stage():
                deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)

    def schedule_b(self, scheduled_node):
        WeightGradStore.set_combine_bw(scheduled_node.type == 'BW')
        if not self.forward_only:
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = [None] * len(self.send_tensor_shapes)
            else:
                output_tensor_grad = self.recv_backward_buffer.pop(0)
                for h in output_tensor_grad[1]:
                    for hh in h:
                        hh.wait()
                output_tensor_grad = output_tensor_grad[0]
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, self.model_type,
                self.config
            )
            self.send_backward_buffer.append(input_tensor_grad)
            WeightGradStore.flush()

    def schedule_w(self, scheduled_node):
        if not self.forward_only:
            WeightGradStore.pop()

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
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.collect_non_loss_data = collect_non_loss_data
        self.forward_only = forward_only
        self._reset()
        self.it = 0



    def run(self):
        schedules = self.get_schedules()
        self.disable_grad_sync()
        for it in range(len(schedules)):
            
            scheduled_node = schedules[it]
            if scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                next_is_comm = it + 1 < len(schedules) and schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], schedules[it + 1:]))
                next_compute = next_compute[0] if len(next_compute) > 0 else None
                self.add_communication(scheduled_node, next_is_comm, next_compute)
            elif scheduled_node.type == 'F':
                self.schedule_f(scheduled_node)
            elif scheduled_node.type in {'B', 'BW'}:
                self.schedule_b(scheduled_node)
            elif scheduled_node.type == 'W':
                self.schedule_w(scheduled_node)
            else:
                raise ValueError(f"Unknown node type {scheduled_node.type}")

        for h in self.send_handles:
            for hh in h:
                for hhh in hh:
                    hhh.wait()

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
        self.prepare(*args, **kwargs)
        return self.run()

zb_scheduler = ZeroBubbleScheduler()

def bootstrap_p2p_communication(config):
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        nccl_init_tensor = [torch.Tensor([0]).cuda()]
        shape = [(1,)]
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_forward(shape, config)
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            send_forward(nccl_init_tensor, shape, config)
            recv_backward(shape, config)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            send_backward(nccl_init_tensor, shape, config)

        torch.distributed.barrier()

def get_zero_bubble_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func_origin, forward_backward_no_pipelining
    if pipeline_model_parallel_size <= 1:
        return forward_backward_no_pipelining
    if get_args().zero_bubble:
        assert not get_args().overlap_grad_reduce
        assert not get_args().sequence_parallel
        assert get_args().transformer_impl == 'local'
        return zb_scheduler
    return zb_scheduler if get_args().zero_bubble else get_forward_backward_func_origin()