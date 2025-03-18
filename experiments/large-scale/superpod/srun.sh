export REPO_PATH="/home/$USER/workspace/PipeMorph"

method_values=("1F1B" "ZB" "ZB-CPU-ReSchedule")

for method in "${method_values[@]}"; do
    export METHOD=$method

    # 2DP, 8TP, 8PP
    export NODES=16
    export GPUS_PER_NODE=8

    srun --account lsdisttrain --partition large --nodes $NODES --gpus-per-node 8 --no-container-mount-home --container-remap-root --container-mounts=/home/$USER/workspace:/home/$USER/workspace --container-workdir=/home/$USER/workspace --container-writable --container-image $LOCAL_IMAGE $REPO_PATH/experiments/large-scale/superpod/test.sh
done