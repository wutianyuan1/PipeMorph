#!/bin/bash

# Define the IP addresses of the nodes
nodes=("172.24.82.221" "172.24.82.222" "172.24.82.223" "172.24.82.224")

# Loop through each node and launch the command with the appropriate R value
for i in "${!nodes[@]}"; do
    node="${nodes[$i]}"
    echo "Launching on node $node with R=$i"
    ssh -i ~/workspace/aliyun.pem $node "/root/miniconda3/bin/torchrun --nnodes 4 --nproc-per-node 1 --node-rank ${i} --master-addr 172.24.82.221 --master-port 9969 /root/workspace/test-varuna/zerobubble/megatron/core/pipeline_parallel/new_delegate.py" &
done

# Wait for all background processes to finish
wait

echo "All nodes have completed their tasks."