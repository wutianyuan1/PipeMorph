#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

void register_pinned_memory(uint64_t mem, size_t nbytes) {
    void* addr = reinterpret_cast<void*>(mem);
    auto ret = cudaHostRegister(addr, nbytes, cudaHostAllocPortable);
    if (ret == cudaSuccess)
        printf("Pinned memory registered\n");
}

void unregister_pinned_memory(uint64_t mem) {
    void* addr = reinterpret_cast<void*>(mem);
    auto ret = cudaHostUnregister(addr);
    if (ret == cudaSuccess)
        printf("Pinned memory unregistered\n");
}

void cudaD2H(at::Tensor& src_, uint64_t dst_, const size_t nbytes) {
    auto src = src_.data_ptr<torch::Half>();
    auto dst = reinterpret_cast<short*>(dst_);
    cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
}

void cudaD2HAsync(at::Tensor& src_, uint64_t dst_, const size_t nbytes) {
    auto src = src_.data_ptr<torch::Half>();
    auto dst = reinterpret_cast<short*>(dst_);
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost);
}

void cudaH2D(uint64_t src_, at::Tensor& dst_, const size_t nbytes) {
    auto dst = dst_.data_ptr<torch::Half>();
    auto src = reinterpret_cast<short*>(src_);
    cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
}

void cudaH2DAsync(uint64_t src_, at::Tensor& dst_, const size_t nbytes) {
    auto dst = dst_.data_ptr<torch::Half>();
    auto src = reinterpret_cast<short*>(src_);
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_pinned_memory", &register_pinned_memory);
    m.def("unregister_pinned_memory", &unregister_pinned_memory);
    m.def("cudaD2H", &cudaD2H);
    m.def("cudaD2HAsync", &cudaD2HAsync);
    m.def("cudaH2D", &cudaH2D);
    m.def("cudaH2DAsync", &cudaH2DAsync);
  }