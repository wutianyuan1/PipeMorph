#!/bin/bash
export REPO_PATH="/root/workspace/PipeMorph-AE"
export NODES=4
export GPUS_PER_NODE=1
# Magic: get the last byte of our IP address, the foure nodes are 172.24.82.[221-224]
LAST_OCTET=$(ip -4 addr show | grep -oP 'inet \K172\.24\.82\.\K\d+' | head -1)
RANK=$((LAST_OCTET - 221))

method_values=("ALL-CPU")

for method in "${method_values[@]}"; do
    export METHOD=$method

    export NUM_DELEGATES=1
    export OUT_DIR="$REPO_PATH/ae/fig20/$METHOD/$slowlink/$delay"
    mkdir -p "$OUT_DIR"
    $REPO_PATH/ae/aliyun-run.sh
done