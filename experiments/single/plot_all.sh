export REPO_PATH="/home/$USER/workspace/PipeMorph"

method_values=("1F1B" "ZB" "ZB-CPU" "ZB-CPU-ReSchedule")
model_values=("60B")
delay_values=("0.03" "0.06")

for method in "${method_values[@]}"; do
    for model in "${model_values[@]}"; do
        export METHOD=$method
        export MODEL=$model
         

        # 1DP, 1TP, 4PP
        if [ "$MODEL" == "7B" ]; then
            NODES=4
            GPUS_PER_NODE=1
            DP_SIZE=1
            TP_SIZE=1
            delay_links=("0_1" "2_3")
        fi

        # 1DP, 1TP, 8PP
        if [ "$MODEL" == "14B" ]; then
            NODES=8
            GPUS_PER_NODE=1
            DP_SIZE=1
            TP_SIZE=1
            delay_links=("0_1" "6_7")
        fi

        # 4DP, 2TP, 8PP
        if [ "$MODEL" == "30B" ]; then
            NODES=8
            GPUS_PER_NODE=8
            DP_SIZE=4
            TP_SIZE=2
            delay_links=("0_1" "6_7")
        fi

        # 2DP, 4TP, 8PP
        if [ "$MODEL" == "60B" ]; then
            NODES=8
            GPUS_PER_NODE=8
            TP_SIZE=4
            DP_SIZE=2
            delay_links=("0_1" "6_7")
        fi
        PP_SIZE=$(( $NODES * $GPUS_PER_NODE / $TP_SIZE / $DP_SIZE ))
        for link in "${delay_links[@]}"; do
            for delay in "${delay_values[@]}"; do
                echo "Plotting ${model}-${method} PP=${PP_SIZE} DP=${DP_SIZE} TP=${TP_SIZE} link=${link} delay=${delay}"
                if [ "$method" == "1F1B" ]; then
                    OUT_DIR=${REPO_PATH}/experiments/single/${model}/${method}/${link}/${delay} python ${REPO_PATH}/zerobubble/utils/plot_1f1b.py --num_stages ${PP_SIZE} --dp_tp_prod $(( $DP_SIZE * $TP_SIZE ))
                else
                    OUT_DIR=${REPO_PATH}/experiments/single/${model}/${method}/${link}/${delay} python ${REPO_PATH}/zerobubble/utils/plot_real.py --num_stages ${PP_SIZE} --dp_tp_prod $(( $DP_SIZE * $TP_SIZE ))
                fi
            done
        done    
    done
done