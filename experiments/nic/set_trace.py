import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=str)
args = parser.parse_args()
user_name = os.getenv("SLURM_JOB_USER")
with open(f"/home/{user_name}/workspace/PipeMorph/zerobubble/megatron/core/failslow_injection/slowlinks.trace", 'w') as f:
    iters = args.iters.split(',')
    f.write('\n'.join([f"{iter};;inf" for iter in iters]))