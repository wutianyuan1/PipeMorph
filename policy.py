import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MockAgentNet(nn.Module):
    def __init__(self, B=10, embedding_dim=4):
        super(MockAgentNet, self).__init__()
        self.embedding = nn.Embedding(3, embedding_dim, padding_idx=2)  # "F"=0, "B"=1, "PAD"=2
        self.fc1 = nn.Linear(4 * B * embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

def encode_queue(queue, max_length):
    mapping = {ForwardBatch: 0, BackwardBatch: 1}
    encoded = [mapping.get(type(x), 2) for x in queue]  # Use 2 for padding
    encoded += [2] * (max_length - len(encoded))
    return encoded

def prepare_input(task_queue, finish_queue, max_length):
    task_encoded = encode_queue(task_queue, max_length)
    finish_encoded = encode_queue(finish_queue, max_length)
    return torch.tensor(task_encoded + finish_encoded, dtype=torch.long)

class LearnedPolicy(nn.Module, PipelinePolicy):
    def __init__(self, num_stages: int, num_batches: int) -> None:
        super(LearnedPolicy, self).__init__()
        self.num_stages = num_stages
        self.num_batches = num_batches
        self.net = MockAgentNet(self.num_batches)

    def pick_batch_to_run(self, task_queue: List[Batch], finish_queue: List[Batch] = None) -> int:
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
