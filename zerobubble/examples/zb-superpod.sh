#!/bin/bash
export USR_HOME=lcaoar
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lcaoar/workspace/hiredis/build
export LD_PRELOAD=/home/$USR_HOME/workspace/PipeMorph/zerobubble/megatron/core/failslow_injection/libinjection.so
export ENABLE_ZERO_BUBBLE=1 
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd /home/$USR_HOME/workspace/PipeMorph/zerobubble
source ~/.bashrc
echo `pwd`
echo $SLURM_NODELIST
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

DATASET="/home/$USR_HOME/workspace/PipeMorph/zerobubble/zb_sample_dataset/dataset/c4_text_document"

if [ ! -e "$DATASET"".idx" ]; then
  wget https://huggingface.co/datasets/ufotalent/zero_bubble_sample_dataset/resolve/main/zb_sample_dataset.tar.gz
  tar -xvf zb_sample_dataset.tar.gz -C /home/$USR_HOME/workspace/PipeMorph/zerobubble/
fi

# Running locally
if [ -z "$WORLD_SIZE" ]; then
  export WORLD_SIZE=4
  export RANK=$PMIX_RANK
  export MASTER_PORT=10086
  export MASTER_ADDR=`python utils/parse_slurm_nodes.py`
fi

if [ -z "$EXIT_INTERVAL" ]; then
  EXIT_INTERVAL=1000
fi

WORLD_SIZE_IN_GPUS=4

if [ -z "$PIPELINE_SIZE" ]; then
  export PIPELINE_SIZE=$(( $WORLD_SIZE_IN_GPUS))
  LAYERS=$(( $PIPELINE_SIZE * 6 - 2))
  MICRO_BATCH_SIZE=1
  GLOBAL_BATCH_SIZE=$(( $PIPELINE_SIZE * 3 * $MICRO_BATCH_SIZE ))
  HIDDEN_SIZE=6144
  ATTENTION_HEADS=32
  ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
fi

profile_ranks="0"
for ((i = 1; i < $WORLD_SIZE_IN_GPUS; i++)); do
    profile_ranks="$profile_ranks $i"
done
if [ -z "$ZERO_BUBBLE_TIMER_START" ]; then
  ZERO_BUBBLE_TIMER_START=100
  ZERO_BUBBLE_TIMER_END=110
fi

if [ -z "$EVAL_INTERVAL" ]; then
  EVAL_INTERVAL=10000
fi

if [ -z "$TP_SIZE" ]; then
  TP_SIZE=1
fi

if [ -z "$TRAIN_SAMPLES" ]; then
  TRAIN_SAMPLES=360
fi

options=" \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PIPELINE_SIZE \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $ATTENTION_HEADS \
  --exit-interval $EXIT_INTERVAL \
  --seq-length 1024 \
  --max-position-embeddings 2048 \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-samples $TRAIN_SAMPLES \
  --lr-decay-samples 126953125 \
  --lr-warmup-samples 183105 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-decay-style cosine \
  --log-interval 1 \
  --eval-iters 1 \
  --eval-interval $EVAL_INTERVAL \
  --data-path ${DATASET} \
  --tokenizer-type GPTSentencePieceTokenizer \
  --tokenizer-model /home/$USR_HOME/workspace/PipeMorph/zerobubble/zb_sample_dataset/tokenizers/tokenizer.model \
  --split 98,2,0 \
  --clip-grad 8.0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.006 \
  --no-barrier-with-level-1-timing \
  --profile-step-start 150 \
  --profile-step-end 170 \
  --profile-ranks $profile_ranks \
  --allow-padding-num-layers"

if [ -z "$FP32" ]; then
  options="$options --fp16"
fi

if [ ! -z "$PROFILED" ]; then
  options="$options --profile"
fi

if [ ! -z "$ZERO_BUBBLE_V_SCHEDULE" ]; then
  ENABLE_ZERO_BUBBLE=1
  options="$options --zero-bubble-v-schedule "
fi

if [ ! -z "$ENABLE_ZERO_BUBBLE" ]; then
  options="$options --enable-zero-bubble \
  --zero-bubble-pipeline-timers-start-iter $ZERO_BUBBLE_TIMER_START \
  --zero-bubble-pipeline-timers-end-iter $ZERO_BUBBLE_TIMER_END \
  --zero-bubble-max-pending-backward $ZERO_BUBBLE_MEM_LIMIT"
fi

if [ ! -z "$ENABLE_EXACTLY_NUMERIC_MATCH" ]; then
  options="$options --enable-exactly-numeric-match \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0"
fi

if [ ! -z "$INTERLEAVED_1F1B" ]; then
  options="$options --num-layers-per-virtual-pipeline-stage 1"
fi

run_cmd="torchrun --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nproc_per_node=$SLURM_GPUS_PER_NODE ${DIR}/pretrain_gpt.py $@ ${options}"

if [ ! -z "$PROFILED" ]; then
  run_cmd="nsys profile -s none -t nvtx,cuda \
    --output $AIP_RUN_NAME.$RANK.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $run_cmd"
fi

if [[ $RANK -eq 0 ]]; then
  echo "Start redis server on RANK 0..."
  redis-server --bind $MASTER_ADDR --save "" --appendonly no &
  REDIS_PID=$!
fi


echo $run_cmd
eval $run_cmd

set +x

if [[ $RANK -eq 0 ]]; then
  kill $REDIS_PID
  python ./utils/plot_real.py --num_stages $PIPELINE_SIZE --dp_tp_prod $TP_SIZE
fi
