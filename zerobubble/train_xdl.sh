yum install -y jemalloc
yum install -y redis
redis-server &
ENABLE_ZERO_BUBBLE=1 ALIBABA_CLUSTER=1 ./examples/pretrain_zero_bubble.sh
#TODO: install jemalloc and redis at pod launch time
#TODO: install apex??
#TODO: the config of autoschedule should match the actual #GPUs.