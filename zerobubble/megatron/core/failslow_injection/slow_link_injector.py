import redis
import time
import threading
from typing import List, Tuple, Union


class SlowLinkInjector(object):
    def __init__(self, trace_path: str, redis_client: redis.StrictRedis):
        self.trace_path = trace_path
        self.redis_client = redis_client
        self.redis_client.set("if_nic_crash", "no")
        self.redis_client.set("slow_links", "")
        self.redis_client.set("sleep_time", "")
        self.trace = []
        with open(self.trace_path, 'r') as f:
            content = f.read().split("\n")
            for line in content:
                if len(line) == 0:
                    continue
                iter_cnt, link, sleep_time = line.split(';')
                self.trace.append([int(iter_cnt), link, sleep_time])
        self.line_no = 0

    def step(self, iteration: int):
        if self.line_no >= len(self.trace):
            return
        if iteration < self.trace[self.line_no][0]:
            return
        elif iteration == self.trace[self.line_no][0]:
            _, link, sleep_time = self.trace[self.line_no]
            if sleep_time != 'inf' and len(link) != 0:
                assert len(link.split(',')) == len(sleep_time.split(','))
                print("!!!!", link, sleep_time)
                self.redis_client.set("slow_links", link)
                self.redis_client.set("sleep_time", sleep_time)
            else:
                print("!!!! NICs crashed")
                assert sleep_time == 'inf'
                self.redis_client.set("if_nic_crash", "yes")
            self.line_no += 1


class SlowLinkInjectorWithTiming(object):
    '''
    Instead of using iteration index as the straggler boundary like `SlowLinkInjector`,
    this injector injects each event when its specific real timing comes.
    '''
    def __init__(self, trace_path: str, redis_client: redis.StrictRedis):
        '''
        Similar to `SlowLinkInjector.__init__`, except parse timing instead of iteration.
        '''
        self.trace_path = trace_path
        self.redis_client = redis_client
        self.redis_client.set("if_nic_crash", "no")
        self.redis_client.set("slow_links", "")
        self.redis_client.set("sleep_time", "")
        self.trace: List[Tuple[float, str, Union[float, str]]] = [] # (timing, link, delay)
        with open(self.trace_path, 'r') as f:
            content = f.read().split("\n")
            for line in content:
                if len(line) == 0:
                    continue
                timing, link, sleep_time = line.split(';')
                self.trace.append((float(timing), link, sleep_time))
                if len(self.trace) > 1:
                    # These events should be injected in the order of time.
                    assert self.trace[-1][0] > self.trace[-2][0]

    def run(self, stop_event: threading.Event):
        t_start = time.time()
        num_injected_events = 0
        while num_injected_events < len(self.trace):
            timing, link, sleep_time = self.trace[num_injected_events]
            # Wait for timing of injecting.
            while time.time() - t_start < timing:
                if stop_event.is_set():
                    print("!!!! Injector stops running")
                    return
                # TODO: Lunxi: Granularity of timing check to be determined.
                time.sleep(0.1)
            if sleep_time != 'inf' and len(link) != 0:
                assert len(link.split(',')) == len(sleep_time.split(','))
                print("!!!!", num_injected_events, link, sleep_time)
                self.redis_client.set("slow_links", link)
                self.redis_client.set("sleep_time", sleep_time)
            else:
                print("!!!! NICs crashed")
                assert sleep_time == 'inf'
                self.redis_client.set("if_nic_crash", "yes")
            num_injected_events += 1
