#!/bin/bash

delay_values=("0.03" "0.06")

if [ "$MODEL" == "7B" ]; then
    export PIPELINE_SIZE=4
    export TP_SIZE=1

    export LAYERS=$(( $PIPELINE_SIZE * 6 - 2))
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $MICRO_BATCH_SIZE ))
    export HIDDEN_SIZE=5120
    export ATTENTION_HEADS=32
    export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

    export TRAIN_SAMPLES=180
    
    slowlink_values=("0_1" "2_3")
fi

if [ "$MODEL" == "14B" ]; then
    export PIPELINE_SIZE=8
    export TP_SIZE=1

    export LAYERS=$(( $PIPELINE_SIZE * 6 - 2))
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $MICRO_BATCH_SIZE ))
    export HIDDEN_SIZE=5120
    export ATTENTION_HEADS=32
    export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

    export TRAIN_SAMPLES=360

    slowlink_values=("0_1" "6_7")
fi

if [ "$MODEL" == "30B" ]; then
    export PIPELINE_SIZE=8
    export TP_SIZE=2

    export LAYERS=$(( $PIPELINE_SIZE * 5 - 2))
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $MICRO_BATCH_SIZE ))
    export HIDDEN_SIZE=8192
    export ATTENTION_HEADS=32
    export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

    export TRAIN_SAMPLES=360

    slowlink_values=("0_1" "6_7")
fi

for delay in "${delay_values[@]}"; do
    if [ "$delay" == "0.03" ]; then
        export NUM_DELEGATES=3
    fi
    if [ "$delay" == "0.06" ]; then
        export NUM_DELEGATES=6
    fi
    for slowlink in "${slowlink_values[@]}"; do
        python $REPO_PATH/experiments/single/set_trace.py --iter 7 --slowlink "$slowlink" --delay_in_sec "$delay"
        export OUT_DIR="$REPO_PATH/experiments/single/$MODEL/$METHOD/$slowlink/$delay"
        mkdir -p "$OUT_DIR"
        $REPO_PATH/zerobubble/examples/zb-superpod.sh
    done
done