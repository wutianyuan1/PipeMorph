#!/bin/bash
if [ -z "$S" ]; then
  S=4
fi
# export LD_PRELOAD=/workspace/test-varuna/zerobubble/megatron/core/failslow_injection/libinjection.so
export ENABLE_ZERO_BUBBLE=1
export GPUS_PER_NODE=$S
./examples/pretrain_zero_bubble.sh
# nsys profile -s none -t nvtx,cuda --output n_delegates-1.nsys-rep --force-overwrite true ./examples/pretrain_zero_bubble.sh
python ./utils/plot_real.py --num_stages $S