export REPO_PATH="/home/$USER/workspace/PipeMorph"

# method_values=("1F1B" "ZB" "ZB-CPU" "ZB-CPU-ReSchedule")
method_values=("1F1B")
model_values=("7B")

for method in "${method_values[@]}"; do
    for model in "${model_values[@]}"; do
        export METHOD=$method
        export MODEL=$model

        # 1DP, 1TP, 4PP
        if [ "$MODEL" == "7B" ]; then
            export NODES=4
            export GPUS_PER_NODE=1
        fi

        # 1DP, 1TP, 8PP
        if [ "$MODEL" == "14B" ]; then
            export NODES=8
            export GPUS_PER_NODE=8
        fi

        # 4DP, 2TP, 8PP
        if [ "$MODEL" == "30B" ]; then
            export NODES=8
            export GPUS_PER_NODE=8
        fi

        # 2DP, 8TP, 4PP
        if [ "$MODEL" == "60B" ]; then
            export NODES=8
            export GPUS_PER_NODE=8
        fi

        if [ $GPUS_PER_NODE -eq 8 ]; then
            PARTITION="large"
        else
            PARTITION="normal"
        fi

        srun --account lsdisttrain --partition $PARTITION --nodes $NODES --gpus-per-node $GPUS_PER_NODE --no-container-mount-home --container-remap-root --container-mounts=/home/$USER/workspace:/home/$USER/workspace --container-workdir=/home/$USER/workspace --container-writable --container-image $LOCAL_IMAGE $REPO_PATH/experiments/seqlen/test.sh
    done
done