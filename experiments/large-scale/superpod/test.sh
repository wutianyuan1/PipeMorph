#!/bin/bash


# 2DP, 8TP, 8PP
export PIPELINE_SIZE=8
export TP_SIZE=8
export DP_SIZE=$(( $NODES * $GPUS_PER_NODE / $TP_SIZE / $PIPELINE_SIZE ))

export LAYERS=$(( $PIPELINE_SIZE * 10 - 2))
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $DP_SIZE * $MICRO_BATCH_SIZE ))
export HIDDEN_SIZE=12288
export ATTENTION_HEADS=32
export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))

export TRAIN_SAMPLES=$(( $GLOBAL_BATCH_SIZE * 170 ))

if [[ $METHOD == "1F1B" ]]; then
  export SEQ_LENGTH=1024
else
  export SEQ_LENGTH=256
fi


export NUM_DELEGATES=6
python $REPO_PATH/experiments/large-scale/superpod/set_trace.py
export OUT_DIR="$REPO_PATH/experiments/large-scale/superpod/$METHOD"
mkdir -p "$OUT_DIR"
# Before running the training job, clear potential shms
df -h /dev/shm
python $REPO_PATH/zerobubble/utils/clear_shm.py
df -h /dev/shm
$REPO_PATH/zerobubble/examples/zb-superpod.sh
