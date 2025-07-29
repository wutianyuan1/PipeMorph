#!/bin/bash
export REPO_PATH="/root/workspace/PipeMorph-AE"
export NODES=4
export GPUS_PER_NODE=1
# Magic: get the last byte of our IP address, the foure nodes are 172.24.82.[221-224]
LAST_OCTET=$(ip -4 addr show | grep -oP 'inet \K172\.24\.82\.\K\d+' | head -1)
RANK=$((LAST_OCTET - 221))

method_values=("ZB-CPU-ReSchedule" "ZB-CPU" "ZB" "1F1B")

for method in "${method_values[@]}"; do
    export METHOD=$method

    delay_values=("0.06" "0.12")
    slowlink_values=("0_1" "2_3")

    for delay in "${delay_values[@]}"; do
        if [ "$delay" == "0.06" ]; then
            export NUM_DELEGATES=3
        fi
        if [ "$delay" == "0.12" ]; then
            export NUM_DELEGATES=6
        fi
        for slowlink in "${slowlink_values[@]}"; do
            if [ "$RANK" -eq 0 ]; then
                echo "$delay sec: NUM_DELEGATES = $NUM_DELEGATES"
                echo "Only rank $RANK sets trace: $delay sec on $slowlink"
                python $REPO_PATH/ae/set_trace.py --iter 7 --slowlink "$slowlink" --delay_in_sec "$delay"
            fi
            export OUT_DIR="$REPO_PATH/ae/fig14_16/$METHOD/$slowlink/$delay"
            mkdir -p "$OUT_DIR"
            $REPO_PATH/ae/aliyun-run.sh
        done
    done
done