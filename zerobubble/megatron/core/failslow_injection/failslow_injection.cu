#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <dlfcn.h>
#include <sys/time.h>
#include <memory>
#include "event_handler.h"

#define CHECK_INTERVAL 64
#define RETRIEVE_NCCL_FUNC(func_name)\
    using func_t = typeof(func_name);\
    auto real_func = reinterpret_cast<func_t*>(dlsym(status->nccl_lib_handle, #func_name));


struct GlobalStatus {
    void* nccl_lib_handle;
    clock_t g_clock_rate = 0;
    EventHandler* event_handler;
    int recv_count = 0;
    float sleep_time = 0.0f;
    std::vector<std::tuple<int, int>> slow_links;
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
    status->event_handler = new EventHandler("localhost", 6379);
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
    // printf("RANK%d send %ld bytes, type=%d\n", std::stoi(getenv("RANK")), count, datatype);
    if (status->recv_count++ % CHECK_INTERVAL == 0) {
        status->sleep_time = status->event_handler->get_sleep_time();
        status->slow_links = status->event_handler->get_slow_links();
    }
    if (status->sleep_time != 0.0f && status->slow_links.size() != 0) {
        int my_rank = std::stoi(getenv("RANK"));
        for (const auto& [start, end] : status->slow_links) {
            if ((my_rank == start && peer == end) || (my_rank == end && peer == start))
                gpu_msleep<<<1, 1, 0, stream>>>(status->sleep_time * 1000.0 / 2.0, status->g_clock_rate);
        }
    }
    using func_t = typeof(ncclSend);
    auto ret = (*real_func)(sendbuff, count, datatype, peer, comm, stream);
    return ret;
}


ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream)
{
    if (!sys_inited) init();
    RETRIEVE_NCCL_FUNC(ncclRecv);
    // printf("RANK%d recv %ld bytes, peer=%d\n", std::stoi(getenv("RANK")), count, peer);
    if (status->recv_count++ % CHECK_INTERVAL == 0) {
        status->sleep_time = status->event_handler->get_sleep_time();
        status->slow_links = status->event_handler->get_slow_links();
    }
    if (status->sleep_time != 0.0f && status->slow_links.size() != 0) {
        int my_rank = std::stoi(getenv("RANK"));
        for (const auto& [start, end] : status->slow_links) {
            if ((my_rank == start && peer == end) || (my_rank == end && peer == start))
                gpu_msleep<<<1, 1, 0, stream>>>(status->sleep_time * 1000.0 / 2.0, status->g_clock_rate);
        }
    }
    auto ret = (*real_func)(recvbuff, count, datatype, peer, comm, stream);
    return ret;
}


ncclResult_t ncclGroupStart()
{
    if (!sys_inited) init();
    // if (std::stoi(getenv("RANK")) == 0)
    //     printf("Group start");
    RETRIEVE_NCCL_FUNC(ncclGroupStart);
    return (*real_func)();
}


ncclResult_t ncclGroupEnd()
{
    if (!sys_inited) init();
    // if (std::stoi(getenv("RANK")) == 0)
    //     printf("Group end");
    RETRIEVE_NCCL_FUNC(ncclGroupEnd);
    return (*real_func)();
}
