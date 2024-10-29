from abc import ABC, abstractmethod
from typing import List
from batches import Batch, ForwardBatch, BackwardBatch, BackwardInputBatch, BackwardWeightBatch, BubbleBatch


class PipelinePolicy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None) -> int:
        pass

# Gpipe: forward has higher priority than backward
class GpipePolicy(PipelinePolicy):
    def __init__(self) -> None:
        super().__init__()

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None) -> int:
        priority_map = {ForwardBatch: 3, BackwardInputBatch: 2, BackwardBatch: 2, BackwardWeightBatch: 1}
        minval, minidx = (float("inf"), float("inf")), -1
        for i, batch in enumerate(task_queue):
            cur = (-priority_map[type(batch)], batch.batch_idx)
            if cur < minval:
                minval = cur
                minidx = i
        return minidx


# PipeDream: 1F1B
class PipeDreamPolicy(PipelinePolicy):
    def __init__(self, num_stages: int) -> None:
        super().__init__()
        self.num_stages = num_stages

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None) -> int:
        assert finish_queue is not None, "1F1B requires a non-null finish queue"
        gpu_mem = 0
        for b in finish_queue:
            if isinstance(b, ForwardBatch):
                gpu_mem += 1
            elif isinstance(b, BackwardBatch):
                gpu_mem -= 1
            elif isinstance(b, BackwardInputBatch) or isinstance(b, BackwardWeightBatch):
                gpu_mem -= 0.5

        priority_map = {ForwardBatch: 1, BackwardWeightBatch: 2, BackwardInputBatch: 3, BackwardBatch: 3}
        minval, minidx = (float("inf"), float("inf")), -1
        for i, batch in enumerate(task_queue):
            cur = (-priority_map[type(batch)], batch.batch_idx)
            if cur < minval:
                minval = cur
                minidx = i
        # If in-memory activations >= num_stages, then we cannot execute subsequent forwards
        if gpu_mem >= self.num_stages and isinstance(task_queue[minidx], ForwardBatch):
            return None
        return minidx

