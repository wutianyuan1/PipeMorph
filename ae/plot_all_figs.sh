#!/bin/bash

# Figures from the test on a single delayed link
figs=("14" "16")

for fig in "${figs[@]}"; do
    echo "Plotting fig $fig"
    python /root/workspace/PipeMorph-AE/ae/fig14_16/plot_fig$fig.py
done

# Fig15 relys on results with 0 delay, i.e., the overhead from fig20,
# so we need to process fig20 first
figs=("17" "20" "15")

for fig in "${figs[@]}"; do
    echo "Plotting fig $fig"
    python /root/workspace/PipeMorph-AE/ae/fig$fig/plot_fig$fig.py
done