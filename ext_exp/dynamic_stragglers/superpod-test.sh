#!/bin/bash

# 1DP, 1TP, 4PP
export PIPELINE_SIZE=4
export TP_SIZE=1
export DP_SIZE=$(( $NODES * $GPUS_PER_NODE / $TP_SIZE / $PIPELINE_SIZE ))

export LAYERS=$(( $PIPELINE_SIZE * 6 - 2))
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $DP_SIZE * $MICRO_BATCH_SIZE ))
export HIDDEN_SIZE=5120
export ATTENTION_HEADS=32
export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

export TRAIN_SAMPLES=$(( $DP_SIZE * 180 ))

method_values=("ZB-CPU-ReSchedule" "ZB-CPU" "ZB" "1F1B")
min_duration_values=("50" "100" "200")

for min_duration in "${min_duration_values[@]}"; do
    for method in "${method_values[@]}"; do
        export METHOD=$method
        export NUM_DELEGATES=6
        export INJECT_DYNAMIC_STRAGGLERS=1
        python $REPO_PATH/ext_exp/dynamic_stragglers/set_trace.py --min_duration $min_duration
        export OUT_DIR="$REPO_PATH/ext_exp/dynamic_stragglers/$min_duration/$METHOD"
        mkdir -p "$OUT_DIR"
        # Before running the training job, clear potential shms
        df -h /dev/shm
        python $REPO_PATH/zerobubble/utils/clear_shm.py
        df -h /dev/shm
        $REPO_PATH/zerobubble/examples/zb-superpod-ext-exp.sh
    done
done
