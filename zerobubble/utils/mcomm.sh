#!/bin/bash
cd /root/workspace/PipeMorph/zerobubble
source ~/.bashrc
echo `pwd`
python_exe=`which python`
if [ -z "$S" ]; then
  S=2
fi
# export LD_PRELOAD=/workspace/PipeMorph/zerobubble/megatron/core/failslow_injection/libinjection.so
export ENABLE_ZERO_BUBBLE=1 
export GPUS_PER_NODE=1 NODERANK=$R
/root/miniconda3/pkgs/nsight-compute-2024.1.1.4-h968f9c8_2/nsight-compute-2024.1.1/host/target-linux-x64/nsys profile -s none -t nvtx,cuda --output 4node_$R.nsys-rep --force-overwrite true ./examples/zb-node1.sh
# bash ./examples/zb-node1.sh
if [[ $R -eq 0 ]]; then
    $python_exe ./utils/plot_real.py --num_stages 4
fi
