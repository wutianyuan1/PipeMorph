#!/bin/bash
if [ -z "$S" ]; then
  S=2
fi
# export LD_PRELOAD=/workspace/test-varuna/zerobubble/megatron/core/failslow_injection/libinjection.so
export ENABLE_ZERO_BUBBLE=1 
export GPUS_PER_NODE=$S NODERANK=$R 
# nsys profile -s none -t nvtx,cuda --output node2_$R.nsys-rep --force-overwrite true ./examples/zb-node1.sh
./examples/zb-node1.sh
if [[ $R -eq 0 ]]; then
    python ./plot_real.py --num_stages 4
fi
