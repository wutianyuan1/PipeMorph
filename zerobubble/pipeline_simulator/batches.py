import matplotlib.patches as patches
from matplotlib.axes import Axes
from abc import ABC


FORWARD_TIME = 10
BACKWARD_ITIME = 10
BACKWARD_WTIME = 10
BACKWARD_TIME = BACKWARD_ITIME + BACKWARD_WTIME
SLOW_FACTOR = 1.5


class Batch(ABC):
    def __init__(self, batch_idx: int, execution_time: float, fail_slow: bool = False, min_begin_time: int = -1) -> None:
        super().__init__()
        self.batch_idx = batch_idx
        self.execution_time = execution_time * SLOW_FACTOR if fail_slow else execution_time
        self.execution_begin = -1
        self.color = None
        self.min_begin_time = min_begin_time

    def plot(self, ax: Axes, stage: float, height: float) -> None:
        x, y = self.execution_begin, stage
        rect = patches.Rectangle((x, y), self.execution_time, height, linewidth=1, edgecolor='black', facecolor=self.color)
        ax.add_patch(rect)
        ax.text(x + self.execution_time / 4, y + height / 2, repr(self))


class BubbleBatch(Batch):
    def __init__(self, batch_idx: int, execution_time: float, fail_slow: bool = False, min_begin_time: int = -1) -> None:
        super().__init__(batch_idx, execution_time, fail_slow, min_begin_time)
        self.color = '#FFFFFF'

    def __repr__(self):
        return f"Z{self.batch_idx}"


class ForwardBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1) -> None:
        super().__init__(batch_idx, FORWARD_TIME, fail_slow, min_begin_time)
        self.color = '#FCCCB3'
        self.type = 'F'

    def __repr__(self):
        return f"F{self.batch_idx}"


class BackwardWeightBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1) -> None:
        super().__init__(batch_idx, BACKWARD_WTIME, fail_slow, min_begin_time)
        self.color = '#FBE7A3'
        self.type = 'W'

    def __repr__(self):
        return f"Bw{self.batch_idx}"


class BackwardInputBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1) -> None:
        super().__init__(batch_idx, BACKWARD_ITIME, fail_slow, min_begin_time)
        self.color = '#CBE4E4'
        self.type = 'B'

    def __repr__(self):
        return f"Bi{self.batch_idx}"


class BackwardBatch(Batch):
    def __init__(self, batch_idx: int, fail_slow: bool = False, min_begin_time: int = -1) -> None:
        super().__init__(batch_idx, BACKWARD_TIME, fail_slow, min_begin_time)
        self.color = '#8FAADC'
        self.type = 'B'

    def __repr__(self):
        return f"B{self.batch_idx}"
