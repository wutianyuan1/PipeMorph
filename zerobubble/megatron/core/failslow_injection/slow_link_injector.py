import redis


class SlowLinkInjector(object):
    def __init__(self, trace_path: str, redis_client: redis.StrictRedis):
        self.trace_path = trace_path
        self.redis_client = redis_client
        self.redis_client.set("slow_links", "")
        self.redis_client.set("sleep_time", 0.0)
        self.trace = []
        with open(self.trace_path, 'r') as f:
            content = f.read().split("\n")
            for line in content:
                if len(line) == 0:
                    continue
                iter_cnt, link, sleep_time = line.split(';')
                self.trace.append([int(iter_cnt), link, float(sleep_time)])
        self.line_no = 0

    def step(self, iteration: int):
        if self.line_no >= len(self.trace):
            return
        if iteration < self.trace[self.line_no][0]:
            return
        elif iteration == self.trace[self.line_no][0]:
            _, link, sleep_time = self.trace[self.line_no]
            print("!!!!", link, sleep_time )
            self.redis_client.set("slow_links", link)
            self.redis_client.set("sleep_time", sleep_time)
            self.line_no += 1
