#!/bin/bash
export LOCAL_IMAGE="/home/twubt/containers/pytorch-23.05.sqsh"
export REPO_PATH="/home/$USER/workspace/PipeMorph-Ext-Exp"
export NODES=4
export GPUS_PER_NODE=1

srun --account lsdisttrain --partition normal --nodes $NODES --gpus-per-node $GPUS_PER_NODE --no-container-mount-home --container-remap-root --container-mounts=/home/$USER/workspace:/home/$USER/workspace --container-workdir=/home/$USER/workspace --container-writable --container-image $LOCAL_IMAGE $REPO_PATH/ext_exp/dynamic_stragglers/superpod-test.sh
