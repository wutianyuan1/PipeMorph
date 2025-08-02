#!/bin/bash
export REPO_PATH="/root/workspace/PipeMorph-Ext-Exp"
export NODES=4
export GPUS_PER_NODE=1
# Magic: get the last byte of our IP address, the foure nodes are 172.24.82.[221-224]
LAST_OCTET=$(ip -4 addr show | grep -oP 'inet \K172\.24\.82\.\K\d+' | head -1)
RANK=$((LAST_OCTET - 221))

method_values=("ZB-CPU-ReSchedule" "ZB-CPU" "ZB" "1F1B")

for method in "${method_values[@]}"; do
    export METHOD=$method
    export NUM_DELEGATES=6
    export INJECT_DYNAMIC_STRAGGLERS=1
    if [ "$RANK" -eq 0 ]; then
        echo "NUM_DELEGATES = $NUM_DELEGATES"
        echo "Only rank $RANK sets trace"
        python $REPO_PATH/ext_exp/dynamic_stragglers/set_trace.py
    fi
    export OUT_DIR="$REPO_PATH/ext_exp/dynamic_stragglers/$METHOD"
    mkdir -p "$OUT_DIR"
    $REPO_PATH/ext_exp/aliyun-run.sh
done