import os
from multiprocessing import shared_memory


g_rank = int(os.getenv("RANK"))
num_delegates = int(os.getenv("NUM_DELEGATES"))
gpus_per_node = int(os.getenv("GPUS_PER_NODE"))
for rank in range(g_rank * gpus_per_node, g_rank * gpus_per_node + gpus_per_node):
    for qid in range(num_delegates):
        for op in ['send_forward', 'send_backward', 'recv_forward', 'recv_backward']:
            try:
                shm = shared_memory.SharedMemory(name=f'{op}_shm_{qid}_{rank}', create=False)
                shm.close()
                shm.unlink()
                print(f"Successfully destroyed shared memory: <{op}_shm_{qid}_{rank}>")
            except Exception as e:
                print(f"Failed to destroy shared memory: <{op}_shm_{qid}_{rank}>, reason: {e}")
