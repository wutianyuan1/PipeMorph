
from multiprocessing import shared_memory
import memcpy
import ctypes
import torch
import numpy as np

NUM_TENSORS = 12
NUM_ELEMENTS = 8
FLOAT16_NBYTES = 2
SIGNAL_NBYTES = 2
HEADER_SIZE = NUM_TENSORS * SIGNAL_NBYTES

gpu_buffer = [torch.zeros((NUM_ELEMENTS), dtype=torch.float16).cuda() for _ in range(NUM_TENSORS)]
gpu_signals = [torch.randn((1,), dtype=torch.float16).cuda() for _ in range(NUM_TENSORS)]
shm = shared_memory.SharedMemory(create=True, size=NUM_TENSORS * (NUM_ELEMENTS * FLOAT16_NBYTES + SIGNAL_NBYTES))
address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
memcpy.register_pinned_memory(address, NUM_TENSORS * (NUM_ELEMENTS * FLOAT16_NBYTES + SIGNAL_NBYTES))

for i in range(NUM_TENSORS):
    print(f"D2H micro-batch {i}")
    gpu_t = torch.randn((NUM_ELEMENTS), dtype=torch.float16).cuda()
    print(gpu_t)
    address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf, HEADER_SIZE + i * NUM_ELEMENTS * FLOAT16_NBYTES))
    memcpy.cudaD2H(gpu_t, address, NUM_ELEMENTS * FLOAT16_NBYTES)

    index = (torch.ones((1,), dtype=torch.float16) * i).cuda()
    print(index)
    address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf, i * SIGNAL_NBYTES))
    memcpy.cudaD2H(index, address, SIGNAL_NBYTES)


    np_arr = np.ndarray((1,), dtype=np.float16, buffer=shm.buf[i * SIGNAL_NBYTES : (i + 1) * SIGNAL_NBYTES])
    index = torch.from_numpy(np_arr)
    print(index)

    np_arr = np.ndarray((NUM_ELEMENTS,), dtype=np.float16, buffer=shm.buf[HEADER_SIZE + i * NUM_ELEMENTS * FLOAT16_NBYTES : HEADER_SIZE + (i + 1) * NUM_ELEMENTS * FLOAT16_NBYTES])
    cpu_t = torch.from_numpy(np_arr)
    print(cpu_t)
    
for i in range(NUM_TENSORS):
    print(f"H2D micro-batch {i}")
    address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf, HEADER_SIZE + i * NUM_ELEMENTS * FLOAT16_NBYTES))
    memcpy.cudaH2D(address, gpu_buffer[i], NUM_ELEMENTS * FLOAT16_NBYTES)
    print(gpu_buffer[i])

    address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf, i * SIGNAL_NBYTES))
    memcpy.cudaH2D(address, gpu_signals[i], SIGNAL_NBYTES)
    print(gpu_signals[i])


memcpy.unregister_pinned_memory(address)
shm.close()