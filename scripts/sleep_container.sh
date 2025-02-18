#!/bin/bash
set +x

QUEUE="alimama_k2_match_llm_h20"
WORLD_SIZE=1
OSS_ACCESS_ID=LTAI5tBYK9dcsovL5zZynVcM
OSS_ACCESS_KEY=lLEXHP4FdFsbq4QKsuGiOudkJv9U8R
OSS_BUCKET=424850
OSS_ENDPOINT=oss-cn-hangzhou-zmf.aliyuncs.com

entry_file="scripts/sleep_container.py"
count_1=""

mdl_args="--queue=${QUEUE} \
        --entry=${entry_file} \
        --worker_count=${WORLD_SIZE}  \
        --file.cluster_file=scripts/cluster.json \
        --oss_access_id=${OSS_ACCESS_ID} \
        --oss_access_key=${OSS_ACCESS_KEY} \
        --oss_bucket=${OSS_BUCKET} \
        --oss_endpoint=${OSS_ENDPOINT} \
        --_NEBULA_MODEL=${SAVE_MODEL} \
        --nebula_model=${SAVE_MODEL} \
        --job_name=train_sppo_test \
        --algo_name=pytorch240 \
        "
if [ -n "${OPENLM_TOKEN}" ]; then
    mdl_args="${mdl_args} --env=OPENLM_TOKEN=${OPENLM_TOKEN}"
fi

if [ -n "${ODPS_PROJECT}" ]; then
    mdl_args="$mdl_args --tables=${ODPS_TABLE} --odps_project=${ODPS_PROJECT}"
fi

nebulactl run mdl --user_params="${args}"  $mdl_args

#-m cProfile -o /data/oss_bucket_0/llm/analysis/ppo_perf.pstats