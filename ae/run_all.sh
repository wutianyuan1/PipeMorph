#!/bin/bash

bash /root/workspace/PipeMorph-AE/ae/pkill_all.sh

test="/root/workspace/PipeMorph-AE/ae/$1/test.sh"

# Define the IP addresses of the nodes
nodes=("172.24.82.222" "172.24.82.223" "172.24.82.224")

# Loop through each node and launch the command
for i in "${!nodes[@]}"; do
    node="${nodes[$i]}"
    echo "Start $test on node $node"
    ssh -tt -i ~/workspace/aliyun.pem $node "bash -il -c '$test'" &
done
# Launch the command on the master node
bash $test &

# Wait for all background processes to finish
wait

echo "All nodes have completed their tasks."