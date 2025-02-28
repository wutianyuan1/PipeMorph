import torch
import numpy as np
import torch.distributed as dist


class ProfileBuffer(object):
    def __init__(self, world_size: int, pp_stages: int) -> None:
        self.world_size = world_size
        self.pp_stages = pp_stages
        self.rank = dist.get_rank()
        self.op2idx = {'F': 0, 'B': 1, 'W': 2}
        self.op_count = [0, 0, 0]
        self.profiles = torch.zeros((3, self.world_size), dtype=torch.float, device='cpu')  # [F, B, W] * WORLD_SIZE

    def reset(self):
        self.op_count = [0, 0, 0]
        self.profiles = torch.zeros((3, self.world_size), dtype=torch.float, device='cpu')  # [F, B, W] * WORLD_SIZE

    def record(self, op_type: str, elapsed_time: float):
        idx = self.op2idx[op_type]
        self.op_count[idx] += 1
        self.profiles[idx, self.rank] = self.profiles[idx, self.rank] * (self.op_count[idx] - 1) / self.op_count[idx] + elapsed_time / self.op_count[idx]

    def get_stage_exec_times(self):
        profiles_gpu = self.profiles.cuda(torch.cuda.current_device())
        dist.all_reduce(profiles_gpu)
        profiles_array = profiles_gpu.cpu().numpy()
        ranks_per_stage = self.world_size // self.pp_stages
        result = np.zeros((3, self.pp_stages))
        for idx in range(3):
            for j in range(self.pp_stages):
                result[idx, j] = np.mean(profiles_array[idx, j * ranks_per_stage: (j + 1) * ranks_per_stage])
        return result
