#!/bin/bash

cmd="pkill -9 python; pkill -9 aliyun-run.sh; pkill -9 -f pipemorph_ext_exp.sh"

# Define the IP addresses of the nodes
nodes=("172.24.82.222" "172.24.82.223" "172.24.82.224")

# Loop through each node and launch the command
for i in "${!nodes[@]}"; do
    node="${nodes[$i]}"
    echo "Running '$cmd' on node $node"
    ssh -i ~/workspace/aliyun.pem $node "$cmd"
done
eval $cmd

# Wait for all background processes to finish
wait

echo "All nodes have completed their tasks."