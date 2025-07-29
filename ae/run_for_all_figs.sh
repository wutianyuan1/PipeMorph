#!/bin/bash

# figs=("14_16" "15" "17" "20")
figs=("20")

for fig in "${figs[@]}"; do
    echo "Running for fig $fig"
    bash /root/workspace/PipeMorph-AE/ae/run_all.sh fig$fig
done