import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int)
parser.add_argument('--slowlink', type=str)
parser.add_argument('--delay_in_sec', type=float)
args = parser.parse_args()

with open(f"/root/workspace/PipeMorph-AE/zerobubble/megatron/core/failslow_injection/slowlinks.trace", 'w') as f:
    f.write(f"{args.iter};{args.slowlink};{args.delay_in_sec}")