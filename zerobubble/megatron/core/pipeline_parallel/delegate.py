import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core.pipeline_parallel.event_timer import EventTimer
from queue import Empty


class CommunicationDelegate(mp.Process):
    def __init__(self, task_queue, completion_queue, dist_info):
        super().__init__()
        self.task_queue = task_queue
        self.completion_queue = completion_queue
        self.dist_info = dist_info
        self.timer = EventTimer()

    def run(self):
        delegate_rank = self.dist_info['delegate_rank']
        delegate_world_size = self.dist_info['delegate_world_size']
        master_addr = os.environ['MASTER_ADDR']
        os.environ = {}
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(self.dist_info['delegate_port'])
        dist.init_process_group(
            backend='nccl',
            init_method="env://",
            rank=delegate_rank,
            world_size=delegate_world_size
        )
        # Init notification to main process
        print(f"[CommunicationDelegate {dist.get_rank()}] Initialized with rank {delegate_rank} out of {delegate_world_size}")
        self.completion_queue.put("init")

        while True:
            try:
                task = self.task_queue.get(timeout=10)
                if task is None:
                    print(f"[CommunicationDelegate {dist.get_rank()}] Received sentinel. Exiting.")
                    break
                self.handle_task(task)
            except Empty:
                continue  # Continue listening for tasks
        dist.destroy_process_group()

    def handle_task(self, task):
        task_type = task['type']
        tensor = task['buffer']
        peer_rank = task['peer_rank']
        task_id = task['task_id']

        t0 = time.time()
        timer_id = self.timer.start(t0)
        if task_type == 'send':
            dist.send(tensor, dst=peer_rank)
            self.completion_queue.put(task_id)

        elif task_type == 'recv':
            dist.recv(tensor, src=peer_rank)
            a = tensor.view(-1)[0].item()  # HACK: We must access this tensor to 'sync' the recv operation
            self.completion_queue.put(task_id)
        else:
            print(f"[CommunicationDelegate {dist.get_rank()}] Unknown task type: {task_type}")
        self.timer.end(timer_id, task_type)


def start_communication_delegate(task_queue, completion_queue, dist_info):
    delegate = CommunicationDelegate(task_queue, completion_queue, dist_info)
    delegate.start()
    return delegate


class DelegateManager(object):
    def __init__(self, dist_info, ndeles: int = 2):
        mp.set_start_method("spawn", force=True)
        self.dist_info = dist_info
        self.num_delegates = ndeles
        self.port = 26010
        self.deles_f, self.tqs_f, self.cqs_f = self._create_delegate_and_queues(ndeles)
        self.deles_b, self.tqs_b, self.cqs_b = self._create_delegate_and_queues(ndeles)
        # Wait for delegate processes to initialize
        for i in range(self.num_delegates):
            assert self.cqs_f[i].get() == 'init'
            assert self.cqs_b[i].get() == 'init'
        self.sfid, self.rfid = 0, 0
        self.sbid, self.rbid = 0, 0
        self.task_to_queue_f = {}
        self.task_to_queue_b = {}

    def _create_delegate_and_queues(self, ndeles):
        delegates, task_queues, completion_queues = [], [], []
        for i in range(ndeles):
            task_queue = mp.Queue()
            completion_queue = mp.Queue()
            self.port += 1
            self.dist_info['delegate_port'] = self.port
            delegate = CommunicationDelegate(task_queue, completion_queue, self.dist_info)
            delegate.start()
            task_queues.append(task_queue)
            completion_queues.append(completion_queue)
            delegates.append(delegate)
        return delegates, task_queues, completion_queues

    def reset(self):
        self.sfid, self.rfid = 0, 0
        self.sbid, self.rbid = 0, 0
        self.task_to_queue_f = {}
        self.task_to_queue_b = {}

    def terminate(self):
        for i in range(self.num_delegates):
            self.tqs_f[i].put(None)
            self.tqs_b[i].put(None)
            self.deles_f[i].join()
            self.deles_b[i].join()

    def schedule_f(self, task):
        if task['type'] == 'send':
            qid = self.sfid % self.num_delegates
            self.sfid += 1
        else:
            qid = self.rfid % self.num_delegates
            self.rfid += 1
        self.task_to_queue_f[task['task_id']] = qid
        self.tqs_f[qid].put(task)

    def schedule_b(self, task):
        if task['type'] == 'send':
            qid = self.sbid % self.num_delegates
            self.sbid += 1
        else:
            qid = self.rbid % self.num_delegates
            self.rbid += 1
        self.task_to_queue_b[task['task_id']] = qid
        self.tqs_b[qid].put(task)

    def wait_f(self, task_id):
        queue_id = self.task_to_queue_f[task_id]
        while True:
            try:
                completed_id = self.cqs_f[queue_id].get_nowait()
                if completed_id == task_id:
                    break
            except Empty:
                pass

    def wait_b(self, task_id):
        queue_id = self.task_to_queue_b[task_id]
        while True:
            try:
                completed_id = self.cqs_b[queue_id].get_nowait()
                if completed_id == task_id:
                    break
            except Empty:
                pass
