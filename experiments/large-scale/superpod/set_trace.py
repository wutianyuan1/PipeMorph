import argparse
import os

parser = argparse.ArgumentParser()
args = parser.parse_args()
user_name = os.getenv("SLURM_JOB_USER")
with open(f"/home/{user_name}/workspace/PipeMorph/zerobubble/megatron/core/failslow_injection/slowlinks.trace", 'w') as f:
    with open(f"/home/{user_name}/workspace/PipeMorph/experiments/large-scale/superpod/large-scale.trace", 'r') as f_trace:
        trace = f_trace.read()
    f.write(trace)