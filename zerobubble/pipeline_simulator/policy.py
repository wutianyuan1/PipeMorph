import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List
from pipeline_simulator.batches import Batch, ForwardBatch, BackwardBatch, BackwardInputBatch, BackwardWeightBatch


class PipelinePolicy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None, time: int = 0, stage: int = 0) -> int:
        pass


# Gpipe: forward has higher priority than backward
class GpipePolicy(PipelinePolicy):
    def __init__(self) -> None:
        super().__init__()

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None, time: int = 0, stage: int = 0) -> int:
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

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None, time: int = 0, stage: int = 0) -> int:
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


# ZeroBubble (ICLR'24)
class ZeroBubblePolicy(PipelinePolicy):
    def __init__(self, num_stages: int) -> None:
        super().__init__()
        self.num_stages = num_stages

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None, time: int = 0, stage: int = 0) -> int:
        assert finish_queue is not None, "1F1B requires a non-null finish queue"
        priority_map = {ForwardBatch: 2, BackwardWeightBatch: 1, BackwardInputBatch: 3}
        minval, minidx = (float("inf"), float("inf")), -1
        for i, batch in enumerate(task_queue):
            cur = (-priority_map[type(batch)], batch.batch_idx)
            if cur < minval:
                minval = cur
                minidx = i

        if minidx == -1 or time < task_queue[minidx].min_begin_time:
            minval = (float("inf"), float("inf"))
            # find bw fw batch to fill the bubble
            for i, batch in enumerate(task_queue):
                if batch.min_begin_time <= time:
                    cur = (-priority_map[type(batch)], batch.batch_idx)
                    if cur < minval:
                        minval = cur
                        minidx = i
            
        return minidx  

# ZeroBubble (ICLR'24)
class FixedPolicy(PipelinePolicy):
    def __init__(self, num_stages: int, file: str = 'schedule.txt') -> None:
        super().__init__()
        self.num_stages = num_stages
        with open(file, 'r') as f:
            content = f.read().split("\n")
        self.content = []
        for line in content:
            self.content.append(line.split(" "))
        self.count = {i: 0 for i in range(num_stages)}

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None, time: int = 0, stage: int = 0) -> int:
        to_exec = self.content[stage][self.count[stage]]
        print("="*30)
        print(f'time:{time}, stage:{stage}, count:{self.count}, to_exec={to_exec}, task_q={[(repr(i), i.min_begin_time) for i in task_queue]}')
        idx_in_table = None
        for idx, item in enumerate(task_queue):
            if to_exec == repr(item) and time > item.min_begin_time - 1:
                print(f"Execute: {item}")
                self.count[stage] += 1
                return idx
        return None


class MockAgentNet(nn.Module):
    def __init__(self, B: int = 10, embedding_dim: int = 4) -> None:
        super(MockAgentNet, self).__init__()
        self.embedding = nn.Embedding(3, embedding_dim, padding_idx=2)  # "F"=0, "B"=1, "PAD"=2
        self.fc1 = nn.Linear(4 * B * embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def encode_queue(queue: List, max_length: int) -> List[int]:
    mapping = {ForwardBatch: 0, BackwardBatch: 1}
    encoded = [mapping.get(type(x), 2) for x in queue]  # Use 2 for padding
    encoded += [2] * (max_length - len(encoded))
    return encoded


def prepare_input(task_queue: List, finish_queue: List, max_length: int) -> torch.Tensor:
    task_encoded = encode_queue(task_queue, max_length)
    finish_encoded = encode_queue(finish_queue, max_length)
    return torch.tensor(task_encoded + finish_encoded, dtype=torch.long)


class LearnedPolicy(nn.Module, PipelinePolicy):
    def __init__(self, num_stages: int, num_batches: int) -> None:
        super(LearnedPolicy, self).__init__()
        self.num_stages = num_stages
        self.num_batches = num_batches
        self.net = MockAgentNet(self.num_batches)

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None, time: int = 0) -> int:
        assert finish_queue is not None, "1F1B requires a non-null finish queue"
        # probs = self.net(prepare_input(task_queue, finish_queue, 2 * self.num_batches).unsqueeze(0)).squeeze()
        probs = torch.rand(3)
        priority_map = {ForwardBatch: probs[0], BackwardBatch: probs[1], BackwardInputBatch: probs[2], BackwardWeightBatch: probs[2] - 0.1}
        print(probs)
        minval, minidx = (float("inf"), float("inf")), -1
        for i, batch in enumerate(task_queue):
            cur = (-priority_map[type(batch)], batch.batch_idx)
            if cur < minval:
                minval = cur
                minidx = i
        return minidx