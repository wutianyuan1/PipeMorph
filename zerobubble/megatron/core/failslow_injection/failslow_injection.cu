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
static int N = 1024;


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
    // size_t size = N * N * sizeof(float);
    // cudaMalloc((void**)&(status->d_A), size);
    // cudaMalloc((void**)&(status->d_B), size);
    // cudaMalloc((void**)&(status->d_C), size);
    // status->event_handler = new EventHandler("localhost", 6379);
    sys_inited = true;
}


__global__ static void gpu_msleep(int ms, clock_t clock_rate)
{
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0)/(clock_rate*1.0) < ms)
        t1 = clock64();
}

__global__ static void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;

    for (int k = 0; k < N; ++k) {
        value += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = value;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream)
{
    if (!sys_inited) init();
    RETRIEVE_NCCL_FUNC(ncclSend);
    int my_rank = std::stoi(getenv("RANK"));
    int peer_rank;
    if (peer == 0)
        peer_rank = my_rank - 1;
    else if (peer == 1)
        peer_rank = my_rank + 1;
    // printf("%d ---> %d\n", my_rank, peer_rank);

    // if (my_rank == 0)
    // printf("[SEND] Original Stream %p\n", (void*)stream);
    // cudaStream_t new_stream = status->stream_pool[0]; //[status->stream_id++ % N_STREAMS];
    // cudaStream_t new_stream = status->stream_pool[0];
    // status->stream_id++;
    cudaStream_t new_stream = stream;
    // cudaStreamCreate(&new_stream);
    // if (my_rank == 0)
    // printf("[SEND] New Stream %p\n", (void*)new_stream);
    ncclResult_t ret;
    auto start = std::get<0>(status->slow_links);
    auto end = std::get<1>(status->slow_links);
    if ((my_rank == start && peer_rank == end) || (my_rank == end && peer_rank == start))
        gpu_msleep<<<1, 1, 0, new_stream>>>(status->sleep_time * 1000.0, status->g_clock_rate);
    ret = (*real_func)(sendbuff, count, datatype, peer, comm, new_stream);
    return ret;
    // if (true) {
    //     auto start = std::get<0>(status->slow_links);
    //     auto end = std::get<1>(status->slow_links);
    //     // dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    //     // dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    //     if ((my_rank == start && peer_rank == end) || (my_rank == end && peer_rank == start))
    //     {
    //         // printf("[SEND] Original Stream %p\n", (void*)stream);
    //         // cudaStream_t new_stream = status->stream_pool[0]; //[status->stream_id++ % N_STREAMS];
    //         // cudaStream_t new_stream = status->stream_pool[status->stream_id++ % N_STREAMS];
    //         cudaStream_t new_stream;
    //         cudaStreamCreate(&new_stream);
    //         // printf("[SEND] New Stream %p\n", (void*)new_stream);
    //         // matrixMulKernel<<<dimGrid, dimBlock, 0, new_stream>>>(status->d_A, status->d_B, status->d_C, N);
    //         // gpu_msleep<<<1, 1, 0, new_stream>>>(status->sleep_time * 1000.0, status->g_clock_rate);
    //         ret = (*real_func)(sendbuff, count, datatype, peer, comm, new_stream);
    //     }
    //     else {
    //         ret = (*real_func)(sendbuff, count, datatype, peer, comm, stream);
    //     }
    // }
    // auto ret = (*real_func)(sendbuff, count, datatype, peer, comm, stream);
}


// ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
//                       int peer, ncclComm_t comm, cudaStream_t stream)
// {
//     if (!sys_inited) init();
//     RETRIEVE_NCCL_FUNC(ncclRecv);
//     int my_rank = std::stoi(getenv("RANK"));
//     int peer_rank;
//     if (peer == 0)
//         peer_rank = my_rank - 1;
//     else if (peer == 1)
//         peer_rank = my_rank + 1;
//     // printf("%d ---> %d\n", peer_rank, my_rank);

//     // printf("RANK%d recv %ld bytes, peer=%d\n", std::stoi(getenv("RANK")), count, peer);
//     // if (status->recv_count++ % CHECK_INTERVAL == 0) {
//     //     status->sleep_time = status->event_handler->get_sleep_time();
//     //     status->slow_links = status->event_handler->get_slow_links();
//     // }
//     // if (status->sleep_time != 0.0f && status->slow_links.size() != 0) {
//     //     int my_rank = std::stoi(getenv("RANK"));
//     //     for (const auto& [start, end] : status->slow_links) {
//     //         if ((my_rank == start && peer == end) || (my_rank == end && peer == start))
//     //             gpu_msleep<<<1, 1, 0, stream>>>(status->sleep_time * 1000.0, status->g_clock_rate);
//     //     }
//     // }
//     // ncclResult_t ret; 
//     // if (status->sleep_time != 0.0f && status->slow_links.size() != 0) {
//     //     int my_rank = std::stoi(getenv("RANK"));
//     //     bool sent = false;
//     //     for (const auto& [start, end] : status->slow_links) {
//     //         // 是要变慢的link，把send、recv buffer发N遍，造成变慢
//     //         if ((my_rank == start && peer == end) || (my_rank == end && peer == start)) {
//     //             sent = true;
//     //             size_t offset = 0;
//     //             while (offset < count) {
//     //                 size_t real_size = (COMM_SIZE < (count - offset) ? COMM_SIZE : (count - offset));
//     //                 ret = (*real_func)((char*)recvbuff + offset, real_size, datatype, peer, comm, stream);
//     //                 offset += real_size;
//     //             }
//     //         }
//     //     }
//     //     // 这个rank不应该被影响，正常发
//     //     if (!sent)
//     //         ret = (*real_func)(recvbuff, count, datatype, peer, comm, stream);
//     // }
//     // else {
//     //     // 要是不需要睡就正常转发
//     //     ret = (*real_func)(recvbuff, count, datatype, peer, comm, stream);
//     // }
//     // auto ret = (*real_func)(recvbuff, count, datatype, peer, comm, stream);
//     ncclResult_t ret;
//     if (true) {
//         auto start = std::get<0>(status->slow_links);
//         auto end = std::get<1>(status->slow_links);
//         if ((my_rank == start && peer_rank == end) || (my_rank == end && peer_rank == start))
//         {
//             // printf("[SEND] Original Stream %p\n", (void*)stream);
//             cudaStream_t new_stream;
//             cudaStreamCreate(&new_stream);
//             // printf("[SEND] New Stream %p\n", (void*)new_stream);
//             ret = (*real_func)(recvbuff, count, datatype, peer, comm, new_stream);
//         }
//         else {
//             ret = (*real_func)(recvbuff, count, datatype, peer, comm, stream);
//         }
//     }
//     return ret;
// }


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
