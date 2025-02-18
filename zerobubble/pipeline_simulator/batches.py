import matplotlib.patches as patches
from matplotlib.axes import Axes
from typing import List
from abc import ABC


FORWARD_TIMES = [28, 36, 36, 35]
BACKWARD_ITIMES = [30, 36, 37, 34]
BACKWARD_WTIMES = [22, 27, 27, 26]
BACKWARD_TIMES = [BACKWARD_ITIMES[i] + BACKWARD_WTIMES[i] for i in range(len(BACKWARD_ITIMES))]
SLOW_FACTORS = [1.5, 1.5, 1.5, 1.5]


def update_times(f: List[int], bi: List[int], bw: List[int], slow_factors: List[int] = None) -> None:
    global FORWARD_TIMES, BACKWARD_ITIMES, BACKWARD_WTIMES, BACKWARD_TIMES, SLOW_FACTORS
    FORWARD_TIMES = f
    BACKWARD_ITIMES = bi
    BACKWARD_WTIMES = bw
    BACKWARD_TIMES = [BACKWARD_ITIMES[i] + BACKWARD_WTIMES[i] for i in range(len(BACKWARD_ITIMES))]
    if slow_factors is not None:
        SLOW_FACTORS = slow_factors


class Batch(ABC):
    def __init__(self, batch_idx: int, execution_time: float, fail_slow: bool = False, min_begin_time: int = -1, stage: int = 0) -> None:
        super().__init__()
        self.batch_idx = batch_idx
        self.execution_time = int(execution_time * SLOW_FACTORS[stage]) if fail_slow else execution_time
        self.execution_begin = -1
        self.color = None
        self.min_begin_time = min_begin_time

    def plot(self, ax: Axes, stage: float, height: float) -> None:
        x, y = self.execution_begin, stage
        rect = patches.Rectangle((x, y), self.execution_time, height, linewidth=1, edgecolor='black', facecolor=self.color)
        ax.add_patch(rect)
        ax.text(x + self.execution_time / 4, y + height / 2, repr(self))


class BubbleBatch(Batch):
    def __init__(self, batch_idx: int, execution_time: float, fail_slow: bool = False, min_begin_time: int = -1, stage: int = 0) -> None:
        super().__init__(batch_idx, execution_time, fail_slow, min_begin_time)
        self.color = '#FFFFFF'

    def __repr__(self):
        return f"Z{self.batch_idx}"


class ForwardBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1, stage: int = 0) -> None:
        super().__init__(batch_idx, FORWARD_TIMES[stage], fail_slow, min_begin_time)
        self.color = '#FCCCB3'
        self.type = 'F'

    def __repr__(self):
        return f"F{self.batch_idx}"


class BackwardWeightBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1, stage: int = 0) -> None:
        super().__init__(batch_idx, BACKWARD_WTIMES[stage], fail_slow, min_begin_time)
        self.color = '#FBE7A3'
        self.type = 'W'

    def __repr__(self):
        return f"W{self.batch_idx}"


class BackwardInputBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1, stage: int = 0) -> None:
        super().__init__(batch_idx, BACKWARD_ITIMES[stage], fail_slow, min_begin_time)
        self.color = '#CBE4E4'
        self.type = 'B'

    def __repr__(self):
        return f"B{self.batch_idx}"


class BackwardBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1, stage: int = 0) -> None:
        super().__init__(batch_idx, BACKWARD_TIMES[stage], fail_slow, min_begin_time)
        self.color = '#8FAADC'
        self.type = 'B'

    def __repr__(self):
        return f"B{self.batch_idx}"
