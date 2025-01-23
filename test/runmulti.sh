export MASTER_ADDR=172.31.59.128
export MASTER_PORT=9969
export NODE_RANK=$R
export RANK=$R
export WORLD_SIZE=2

python sendrecv.py
