export METHOD=$1
export REPO_PATH="/home/lcaoar/workspace/PipeMorph"

method_values=("ZB-CPU-ReSchedule" "ZB-CPU" "ZB")
model_values=("7B" "14B" "30B")

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
        srun --account rlsys --nodes $NODES --gpus-per-node $GPUS_PER_NODE --no-container-mount-home --container-remap-root --container-mounts=/home/lcaoar/workspace:/home/lcaoar/workspace --container-workdir=/home/lcaoar/workspace --container-writable --container-image $LOCAL_IMAGE $REPO_PATH/experiments/single/test.sh
    done
done