#!/bin/bash

# 1DP, 1TP, 4PP
if [ "$MODEL" == "7B" ]; then
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
fi

# 1DP, 1TP, 8PP
if [ "$MODEL" == "14B" ]; then
    export PIPELINE_SIZE=8
    export TP_SIZE=1
    export DP_SIZE=$(( $NODES * $GPUS_PER_NODE / $TP_SIZE / $PIPELINE_SIZE ))

    export LAYERS=$(( $PIPELINE_SIZE * 6 - 2))
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $DP_SIZE * $MICRO_BATCH_SIZE ))
    export HIDDEN_SIZE=5120
    export ATTENTION_HEADS=32
    export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

    export TRAIN_SAMPLES=$(( $DP_SIZE * 360 ))
fi

# 4DP, 2TP, 8PP
if [ "$MODEL" == "30B" ]; then
    export PIPELINE_SIZE=8
    export TP_SIZE=2
    export DP_SIZE=$(( $NODES * $GPUS_PER_NODE / $TP_SIZE / $PIPELINE_SIZE ))

    export LAYERS=$(( $PIPELINE_SIZE * 5 - 2))
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $DP_SIZE * $MICRO_BATCH_SIZE ))
    export HIDDEN_SIZE=8192
    export ATTENTION_HEADS=32
    export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

    export TRAIN_SAMPLES=$(( $DP_SIZE * 360 ))
fi

# 2DP, 4TP, 8PP
if [ "$MODEL" == "60B" ]; then
    export PIPELINE_SIZE=8
    export TP_SIZE=4
    export DP_SIZE=$(( $NODES * $GPUS_PER_NODE / $TP_SIZE / $PIPELINE_SIZE ))

    export LAYERS=$(( $PIPELINE_SIZE * 5 - 2))
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $DP_SIZE * $MICRO_BATCH_SIZE ))
    export HIDDEN_SIZE=11264
    export ATTENTION_HEADS=32
    export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

    export TRAIN_SAMPLES=$(( $DP_SIZE * 360 ))
fi

export NUM_DELEGATES=1
export OUT_DIR="$REPO_PATH/experiments/overhead/$MODEL/$METHOD/"
mkdir -p "$OUT_DIR"
$REPO_PATH/zerobubble/examples/zb-superpod.sh
