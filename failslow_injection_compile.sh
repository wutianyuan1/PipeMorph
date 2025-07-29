#!/bin/bash
cd zerobubble/megatron/core/failslow_injection
nvcc -I/root/workspace/hiredis/include -I/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/include/ -L/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/ -L/root/miniconda3/targets/x86_64-linux/lib -I/root/workspace/ -L/root/workspace/hiredis/build -arch=sm_86 -shared -Xcompiler=-fPIC failslow_injection.cu -o libinjection.so -lcuda -lnccl -lcudart -lhiredis
cd ../../../..