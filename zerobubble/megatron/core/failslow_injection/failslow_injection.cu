#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <dlfcn.h>
#include <sys/time.h>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <sstream>
#include "client.h"

#define CHECK_INTERVAL 64
#define RETRIEVE_NCCL_FUNC(func_name)\
    using func_t = typeof(func_name);\
    auto real_func = reinterpret_cast<func_t*>(dlsym(status->nccl_lib_handle, #func_name));


struct GlobalStatus {
    clock_t g_clock_rate;
    float sleep_time;
    int send_count;
    void* nccl_lib_handle;
    Client* redis_client;
    std::vector<std::tuple<int, int>> slow_links;
    int pp_stage;
};

static bool sys_inited = false;
static GlobalStatus* status;


void init() {
    status = new GlobalStatus();
    char* nccl_lib_path = getenv("NCCL_LIB_PATH");
    if (!nccl_lib_path)
        nccl_lib_path = (char*)"/usr/lib/x86_64-linux-gnu/libnccl.so";
    status->nccl_lib_handle = dlopen(nccl_lib_path, RTLD_LAZY);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    status->g_clock_rate = prop.clockRate;
    sys_inited = true;
    status->send_count = 0;
    status->sleep_time = 0.0;
    std::string addr = std::string(getenv("MASTER_ADDR"));
    char* port_str = getenv("REDIS_PORT");
    if (!port_str)
        port_str = (char*)"6379";
    int redis_port = std::atoi(port_str);
    status->redis_client = new Client(addr, redis_port);
    int gpus_per_pp_stage = int(std::atoi(getenv("WORLD_SIZE")) / std::atoi(getenv("PIPELINE_SIZE")));
    status->pp_stage = int(std::atoi(getenv("RANK")) / gpus_per_pp_stage);
    printf("[Injector] Init: my pp_stage = %d, gpus_per_pp_stage = %d\n", status->pp_stage, gpus_per_pp_stage);
}


__global__ static void gpu_msleep(int ms, clock_t clock_rate) {
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0)/(clock_rate*1.0) < ms)
        t1 = clock64();
}


ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    // Init the global status and get the real NCCL function.
    if (!sys_inited) init();
    RETRIEVE_NCCL_FUNC(ncclSend);

    // Check the sleep time and slow links every CHECK_INTERVAL send operations.
    if (status->send_count == 0) {
        status->sleep_time = status->redis_client->get_sleep_time();
        status->slow_links = status->redis_client->get_slow_links();
    }
    status->send_count = (status->send_count + 1) % CHECK_INTERVAL;

    // Check if this link is slow, if it is slow, inject a sleep kernel before calling the real NCCL send.
    int peer_stage = peer;
    if (peer == 0)
        peer_stage = status->pp_stage - 1;
    else if (peer == 1)
        peer_stage = status->pp_stage + 1;
    for (int i = 0; i < status->slow_links.size(); i++) {
        auto start = std::get<0>(status->slow_links[i]);
        auto end = std::get<1>(status->slow_links[i]);
        if ((status->pp_stage == start && peer_stage == end) || (status->pp_stage == end && peer_stage == start))
            gpu_msleep<<<1, 1, 0, stream>>>(status->sleep_time * 1000.0, status->g_clock_rate);
    }

    // Call the real NCCL function to send the data.
    return (*real_func)(sendbuff, count, datatype, peer, comm, stream);
}
