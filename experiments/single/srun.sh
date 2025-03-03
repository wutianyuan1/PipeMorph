export METHOD=$1
export REPO_PATH="/home/$USER/workspace/PipeMorph"

method_values=("1F1B")
model_values=("30B")

for method in "${method_values[@]}"; do
    for model in "${model_values[@]}"; do
        export METHOD=$method
        export MODEL=$model

        if [ "$MODEL" == "7B" ]; then
            NODES=4
            GPUS_PER_NODE=1
        fi

        if [ "$MODEL" == "14B" ]; then
            NODES=8
            GPUS_PER_NODE=1
        fi

        if [ "$MODEL" == "30B" ]; then
            NODES=8
            GPUS_PER_NODE=2
        fi
        srun --account lsdisttrain --nodes $NODES --gpus-per-node $GPUS_PER_NODE --no-container-mount-home --container-remap-root --container-mounts=/home/$USER/workspace:/home/$USER/workspace --container-workdir=/home/$USER/workspace --container-writable --container-image $LOCAL_IMAGE $REPO_PATH/experiments/single/test.sh
    done
done