// DevicePolyMatrix and DeviceDB constructor/destructor implementations.
// Separate .cu file so CUDA-aware malloc/free are available.

#include "types.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

static void cuda_check(cudaError_t err, const char* op) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(err));
    }
}

// ── DevicePolyMatrix ─────────────────────────────────────────────────────────

DevicePolyMatrix::DevicePolyMatrix(uint32_t r, uint32_t c)
    : rows(r), cols(c)
{
    if (r > 0 && c > 0) {
        cuda_check(cudaMalloc(&d_data, byte_size()), "cudaMalloc(DevicePolyMatrix)");
    }
}

DevicePolyMatrix::~DevicePolyMatrix() {
    if (d_data) { cudaFree(d_data); d_data = nullptr; }
}

DevicePolyMatrix::DevicePolyMatrix(DevicePolyMatrix&& other) noexcept
    : d_data(other.d_data), rows(other.rows), cols(other.cols)
{
    other.d_data = nullptr;
    other.rows   = 0;
    other.cols   = 0;
}

DevicePolyMatrix& DevicePolyMatrix::operator=(DevicePolyMatrix&& other) noexcept {
    if (this != &other) {
        if (d_data) cudaFree(d_data);
        d_data       = other.d_data;
        rows         = other.rows;
        cols         = other.cols;
        other.d_data = nullptr;
        other.rows   = 0;
        other.cols   = 0;
    }
    return *this;
}

void DevicePolyMatrix::upload(const uint64_t* host_data, size_t n, cudaStream_t stream) {
    size_t w = (n == 0) ? num_words() : n;
    if (stream == 0) {
        cuda_check(cudaMemcpy(d_data, host_data, w * sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(upload)");
    } else {
        cuda_check(cudaMemcpyAsync(d_data, host_data, w * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(upload)");
    }
}

void DevicePolyMatrix::download(uint64_t* host_data, size_t n, cudaStream_t stream) const {
    size_t w = (n == 0) ? num_words() : n;
    if (stream == 0) {
        cuda_check(cudaMemcpy(host_data, d_data, w * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(download)");
    } else {
        cuda_check(cudaMemcpyAsync(host_data, d_data, w * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(download)");
    }
}

// ── DeviceDB ─────────────────────────────────────────────────────────────────

DeviceDB::~DeviceDB() {
    if (d_data) { cudaFree(d_data); d_data = nullptr; }
}

DeviceDB::DeviceDB(DeviceDB&& other) noexcept
    : d_data(other.d_data), num_words(other.num_words)
{
    other.d_data    = nullptr;
    other.num_words = 0;
}

DeviceDB& DeviceDB::operator=(DeviceDB&& other) noexcept {
    if (this != &other) {
        if (d_data) cudaFree(d_data);
        d_data          = other.d_data;
        num_words       = other.num_words;
        other.d_data    = nullptr;
        other.num_words = 0;
    }
    return *this;
}
