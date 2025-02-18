#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <dlfcn.h>
#include <sys/time.h>
#include <memory>
// #include "event_handler.h"
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <sstream>

#define CHECK_INTERVAL 64
#define RETRIEVE_NCCL_FUNC(func_name)\
    using func_t = typeof(func_name);\
    auto real_func = reinterpret_cast<func_t*>(dlsym(status->nccl_lib_handle, #func_name));
#define COMM_SIZE 512
#define N_STREAMS 32
#define TILE_SIZE 1


struct GlobalStatus {
    void* nccl_lib_handle;
    clock_t g_clock_rate = 0;
    // EventHandler* event_handler;
    int recv_count = 0;
    float sleep_time = 0.040;
    std::tuple<int, int> slow_links {1, 2};
    std::vector<cudaStream_t> stream_pool;
    int stream_id;
    float *d_A, *d_B, *d_C;
};

static bool sys_inited = false;
static GlobalStatus* status;


void init()
{
    status = new GlobalStatus();
    status->nccl_lib_handle = dlopen("/usr/lib/x86_64-linux-gnu/libnccl.so", RTLD_LAZY);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    status->g_clock_rate = prop.clockRate;
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStream_t new_stream;
        cudaStreamCreate(&new_stream);
        status->stream_pool.push_back(new_stream);
    }
    sys_inited = true;
}


__global__ static void gpu_msleep(int ms, clock_t clock_rate)
{
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0)/(clock_rate*1.0) < ms)
        t1 = clock64();
}


ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream)
{
    if (!sys_inited) init();
    RETRIEVE_NCCL_FUNC(ncclSend);
    int my_rank = std::stoi(getenv("RANK"));
    int peer_rank = peer;
    if (peer == 0)
        peer_rank = my_rank - 1;
    else if (peer == 1)
        peer_rank = my_rank + 1;
    auto start = std::get<0>(status->slow_links);
    auto end = std::get<1>(status->slow_links);
    if ((my_rank == start && peer_rank == end) || (my_rank == end && peer_rank == start))
        gpu_msleep<<<1, 1, 0, stream>>>(status->sleep_time * 1000.0, status->g_clock_rate);
    return (*real_func)(sendbuff, count, datatype, peer, comm, stream);
}


ncclResult_t ncclGroupStart()
{
    if (!sys_inited) init();
    RETRIEVE_NCCL_FUNC(ncclGroupStart);
    return (*real_func)();
}


ncclResult_t ncclGroupEnd()
{
    if (!sys_inited) init();
    RETRIEVE_NCCL_FUNC(ncclGroupEnd);
    return (*real_func)();
}
