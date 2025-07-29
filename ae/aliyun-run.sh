#!/bin/bash
export NCCL_SOCKET_IFNAME=eth0
# export NODES=4
# export GPUS_PER_NODE=1
# export NUM_DELEGATES=3
# export METHOD='ZB-CPU-ReSchedule'

# Running locally
export WORLD_SIZE=4
export MASTER_PORT=10086
export MASTER_ADDR='172.24.82.221'
export REDIS_PORT=9969
# Magic: get the last byte of our IP address, the foure nodes are 172.24.82.[221-224]
LAST_OCTET=$(ip -4 addr show | grep -oP 'inet \K172\.24\.82\.\K\d+' | head -1)
export RANK=$((LAST_OCTET - 221))

export USR_HOME=root
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$USR_HOME/workspace/hiredis/build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$USR_HOME/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_PRELOAD=/$USR_HOME/workspace/PipeMorph-AE/zerobubble/megatron/core/failslow_injection/libinjection.so

cd /root/workspace/PipeMorph-AE
python ./zerobubble/utils/clear_shm.py
./zerobubble/examples/pipemorph_ae.sh
