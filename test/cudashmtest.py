import torch
import torch.multiprocessing as mp
import pycuda.driver as cuda
import pycuda.autoinit  # This automatically initializes CUDA
import time
import ctypes

def producer(queue):
    # Initialize CUDA context (handled by pycuda.autoinit)
    
    # Create a CUDA tensor
    in1 = torch.randn((5, 5), device='cuda')
    in1.requires_grad_()
    tensor = in1.pow(2)
    print(f"Producer: Original tensor:\n{tensor}")

    # Send the handle and size to the consumer
    queue.put({'tensorid': 2333, 'data': tensor})
    print("Producer: Tensor sent to consumer.")

    time.sleep(2)
    print(f"Producer: Tensor now: {tensor}")

    time.sleep(10)

def consumer(queue):
    # Receive the IPC handle and size from the producer
    info = queue.get()
    tensor = info['data']
    tid = info['tensorid']
    print(f"Consumer: Received tensor {tid}:\n{tensor}")

    # Perform any operations on the shared tensor
    # tensor += 1
    print(f"Consumer: Modified tensor:\n{tensor}")


def main():
    # Set multiprocessing to use 'spawn' to ensure CUDA context is correctly handled
    mp.set_start_method('spawn')

    # Create a multiprocessing Queue for IPC
    queue = mp.Queue()

    # Create producer and consumer processes
    p_producer = mp.Process(target=producer, args=(queue,))
    p_consumer = mp.Process(target=consumer, args=(queue,))

    # Start the processes
    p_producer.start()
    p_consumer.start()

    # Wait for processes to finish
    p_producer.join()
    p_consumer.join()

if __name__ == '__main__':
    main()
