#!/bin/bash
# cd ../zerobubble/megatron/core/failslow_injection/
# nvcc -I/usr/local/cuda-12.2/targets/x86_64-linux/include/ -L/usr/local/cuda-12.2/targets/x86_64-linux/lib/ -arch=sm_86 -shared failslow_injection.cu -o libinjection.so -lcuda -lnccl -lcudart
# cd ../../../../test/
# export LD_PRELOAD=/workspace/PipeMorph/zerobubble/megatron/core/failslow_injection/libinjection.so
torchrun --nnodes=1 --nproc_per_node=2 sendrecv.py
# python plottest.py