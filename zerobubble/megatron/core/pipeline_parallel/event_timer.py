import torch
import time
from typing import Callable


class TimerItem(object):
    def __init__(self, t_start: float, prompt: str = "Timer", eager_sync: bool = True) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.t_start = t_start
        self.prompt = prompt
        self.eager_sync = eager_sync
        self.sync_done = False
        self.cpu_start = 0.0
        self.cpu_end = 0.0

    def start(self) -> None:
        self.cpu_start = time.time()
        self.start_event.record(stream=torch.cuda.current_stream())

    def end(self) -> None:
        self.end_event.record(stream=torch.cuda.current_stream())
        if self.eager_sync:
            self.end_event.synchronize()
            self.sync_done = True
        self.cpu_end = time.time()

    def elapsed_time(self) -> float:
        if not self.sync_done:
            self.end_event.synchronize()
            self.sync_done = True
        return self.start_event.elapsed_time(self.end_event)

    def print(self, print_func: Callable) -> None:
        print_func(f"{self.prompt} @ {(self.cpu_start * 1000 - self.t_start):4} (ms) to {(self.cpu_end * 1000 - self.t_start):4} (ms), cudaEventElapsed = {self.elapsed_time()} (ms)")


class EventTimer(object):
    def __init__(self) -> None:
        self.timer_items = []

    def start(self, t_start: float, eager_sync: bool = True) -> int:
        # Returns the timer item ID in the list
        timer_item = TimerItem(t_start, eager_sync=eager_sync)
        timer_item.start()
        self.timer_items.append(timer_item)
        return len(self.timer_items) - 1

    def end(self, timer_id: int, prompt: str = "Timer") -> None:
        self.timer_items[timer_id].end()
        self.timer_items[timer_id].prompt = prompt

    def print_all(self, print_func: Callable):
        for item in self.timer_items:
            item.print(print_func)
        self.timer_items = []
