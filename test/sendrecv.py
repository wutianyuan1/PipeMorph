import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from queue import Empty
import multiprocessing

# Ensure CuPy is installed. If not, install it using: pip install cupy-cuda11x
import cupy as cp

# ------------------------------
# CommunicationDelegate Class
# ------------------------------
class CommunicationDelegate(mp.Process):
    def __init__(self, task_queue, completion_queue, shared_memory_info):
        super().__init__()
        self.task_queue = task_queue
        self.completion_queue = completion_queue
        self.shared_memory_info = shared_memory_info
        self.send_buffer = None
        self.recv_buffer = None
        self.send_handle = None
        self.recv_handle = None

    def run(self):
        # Initialize a separate process group for the CommunicationDelegate
        # Assign unique ranks to avoid conflicts with main processes
        delegate_rank = self.shared_memory_info['delegate_rank']
        delegate_world_size = self.shared_memory_info['delegate_world_size']
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=delegate_rank,
            world_size=delegate_world_size
        )
        print(f"[CommunicationDelegate] Initialized with rank {delegate_rank} out of {delegate_world_size}")
        while True:
            try:
                task = self.task_queue.get(timeout=10)  # Adjust timeout as needed
                if task is None:
                    print("[CommunicationDelegate] Received sentinel. Exiting.")
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

        if task_type == 'send':
            print(f"[CommunicationDelegate] Sending tensor {task_id}[sum={torch.sum(tensor)}] to rank {peer_rank}")
            dist.send(tensor, dst=peer_rank)
            print(f"[CommunicationDelegate] Sent tensor {task_id} to rank {peer_rank}")
            self.completion_queue.put(task_id)

        elif task_type == 'recv':
            print(f"[CommunicationDelegate] Receiving tensor {task_id}[before sum={torch.sum(tensor)}] from rank {peer_rank}")
            dist.recv(tensor, src=peer_rank)
            print(f"[CommunicationDelegate] Received tensor {task_id}[after sum={torch.sum(tensor)}] from rank {peer_rank}")
            self.completion_queue.put(task_id)
        else:
            print(f"[CommunicationDelegate] Unknown task type: {task_type}")

# ------------------------------
# Model Definition
# ------------------------------
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(10):
            x = self.fc1(x)
            x = self.relu(x)
        return x

# ------------------------------
# Timer Functions
# ------------------------------
def start_a_timer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream=torch.cuda.current_stream())
    return start, end

def elapsed_time(start: torch.cuda.Event, end: torch.cuda.Event):
    end.record(stream=torch.cuda.current_stream())
    end.synchronize()
    return start.elapsed_time(end)

# ------------------------------
# Communication Delegate Starter
# ------------------------------
def start_communication_delegate(task_queue, completion_queue, shared_memory_info):
    delegate = CommunicationDelegate(task_queue, completion_queue, shared_memory_info)
    delegate.start()
    return delegate

# ------------------------------
# Modified run_pipeline Function
# ------------------------------
def run_pipeline(model, device, rank, execution_sequence, logfile, task_queue, completion_queue, sync=False, repeat_times=1):
    dist.barrier()
    torch.cuda.synchronize()
    
    send_buffers = []
    send_task_ids = []
    recv_buffers = []
    recv_task_ids = []
    
    batch_size = 4096
    input_features = 4096
    input_data_list = [torch.randn(batch_size, input_features).to(device) for _ in range(30)] if rank == 0 else None
    recvs = [torch.zeros(batch_size, input_features).to(device) for _ in range(30)]
    
    t0 = torch.tensor([time.time()], device=device, dtype=torch.float64)
    dist.all_reduce(t0, op=dist.ReduceOp.MAX)
    t0 = t0.cpu().item()
    
    # Define CUDA streams
    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    comp_stream = torch.cuda.Stream(device=device, priority=-100)
    
    for current_step, operation in enumerate(execution_sequence):
        if rank == 0:
            if operation == 'F':
                with torch.cuda.stream(comp_stream):
                    t1 = time.time()
                    s1, e1 = start_a_timer()
                    output = model(input_data_list[current_step // 2])
                    send_buffers.append(output)
                    t_f = elapsed_time(s1, e1)
                    print(f"[Rank{rank}] F @ {(t1 - t0) * 1000:.2f} ms to {(time.time() - t0) * 1000:.2f} ms, cudaEventElapsed = {t_f:.2f} ms", file=logfile, flush=True)
            
            elif operation == 'SEND_FORWARD':
                with torch.cuda.stream(comm_stream):
                    t1 = time.time()
                    s2, e2 = start_a_timer()
                    send_buffer = send_buffers.pop(0)
                    print(f"[Rank{rank}] SEND_FORWARD - Buffer Shape: {send_buffer.shape}", file=logfile, flush=True)

                    dist.send(send_buffer, 1)
                    
                    # # Create a unique task_id
                    # task_id = f'send_{current_step}'
                    
                    # # Create the task dictionary
                    # task = {
                    #     'type': 'send',
                    #     'buffer': send_buffer,
                    #     'peer_rank': 1,
                    #     'task_id': task_id
                    # }
                    
                    # # Enqueue the send task
                    # task_queue.put(task)
                    # send_task_ids.append(task_id)
                    
                    # Record elapsed time
                    t_sf = elapsed_time(s2, e2)
                    print(f"[Rank{rank}] SEND_FORWARD @ {(t1 - t0) * 1000:.2f} ms to {(time.time() - t0) * 1000:.2f} ms, cudaEventElapsed = {t_sf:.2f} ms, Buffer Shape: {send_buffer.shape}", file=logfile, flush=True)
        
        elif rank ==1:
            if operation == 'RECV_FORWARD':
                with torch.cuda.stream(comm_stream):
                    t1 = time.time()
                    s1, e1 = start_a_timer()
                    recv_buffer = recvs[current_step]
                    recv_buffers.append(recv_buffer)

                    dist.recv(recv_buffer, 0)
                    print(recv_buffer)
                    
                    # # Create a unique task_id
                    # task_id = f'recv_{current_step}'
                    
                    # # Create the task dictionary
                    # task = {
                    #     'type': 'recv',
                    #     'buffer': recv_buffer,
                    #     'peer_rank': 0,
                    #     'task_id': task_id
                    # }
                    
                    # # Enqueue the recv task
                    # task_queue.put(task)
                    # recv_task_ids.append(task_id)
                    
                    # Record elapsed time
                    t_rf = elapsed_time(s1, e1)
                    print(f"[Rank{rank}] RECV_FORWARD @ {(t1 - t0) * 1000:.2f} ms to {(time.time() - t0) * 1000:.2f} ms, cudaEventElapsed = {t_rf:.2f} ms", file=logfile, flush=True)
            
            elif operation == 'F':
                with torch.cuda.stream(comp_stream):
                    t2 = time.time()
                    s2, e2 = start_a_timer()
                    
                    # Wait for the corresponding recv task to complete
                    # task_id = recv_task_ids.pop(0)
                    # while True:
                    #     try:
                    #         completed_id = completion_queue.get_nowait()
                    #         if completed_id == task_id:
                    #             break
                    #     except Empty:
                    #         torch.cuda.synchronize()
                    
                    recv_buffer = recvs[current_step]
                    t_wait = elapsed_time(s2, e2)
                    print(f"[Rank{rank}] WAIT @ {(t2 - t0) * 1000:.2f} ms to {(time.time() - t0) * 1000:.2f} ms, cudaEventElapsed = {t_wait:.2f} ms, Buffer Shape: {recv_buffer.shape}", file=logfile, flush=True)
                    
                    # Perform the forward pass using the received buffer
                    t3 = time.time()
                    s3, e3 = start_a_timer()
                    output = model(recv_buffer)
                    t_f = elapsed_time(s3, e3)
                    print(f"[Rank{rank}] F @ {(t3 - t0) * 1000:.2f} ms to {(time.time() - t0) * 1000:.2f} ms, cudaEventElapsed = {t_f:.2f} ms", file=logfile, flush=True)
    
# ------------------------------
# Main Function
# ------------------------------
def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    multiprocessing.set_start_method('spawn')
    
    # Initialize the main process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    print(f"Rank {rank} initialized on device {device}")
    
    # Initialize multiprocessing manager for queues
    # manager = mp.Manager()
    task_queue = mp.Queue()
    completion_queue = mp.Queue()
    
    # Define shared memory info
    shared_memory_info = {
        'send_shape': [4096, 4096],
        'recv_shape': [4096, 4096],
        'delegate_rank': rank,
        'delegate_world_size': 2
    }
    
    # Start the CommunicationDelegate
    delegate = start_communication_delegate(task_queue, completion_queue, shared_memory_info)
    print(f"Started CommunicationDelegate with rank {shared_memory_info['delegate_rank']}")
    
    # Define the execution sequence
    num_steps = 12
    if rank ==0:
        execution_sequence = ['F', 'SEND_FORWARD'] * num_steps
    elif rank ==1:
        execution_sequence = ['RECV_FORWARD', 'F'] * num_steps
    else:
        # For ranks beyond 0 and 1, define behavior or exit
        execution_sequence = []
    
    # Open log file for writing
    logfile = open(f"log_{rank}.log", 'w')
    
    # Construct the model
    print("Constructing model...")
    model = Model().to(device)
    model.train()
    print("Model constructed")
    
    # Number of pipeline iterations
    N = 3
    for i in range(N):
        run_pipeline(
            model=model,
            device=device,
            rank=rank,
            execution_sequence=execution_sequence,
            logfile=(logfile if i == N-1 else open(os.devnull, 'w')),
            task_queue=task_queue,
            completion_queue=completion_queue,
            sync=(i == N - 1)  # Only synchronize on the last iteration
        )
    
    # Cleanup
    task_queue.put(None)  # Sentinel to signal delegate to terminate
    delegate.join()
    dist.destroy_process_group()
    logfile.close()
    print(f"Rank {rank} finished execution and cleaned up.")

if __name__ == "__main__":
    main()
