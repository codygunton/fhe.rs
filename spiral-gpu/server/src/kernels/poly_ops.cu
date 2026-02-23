// NTT-domain polynomial matrix operations — STUB
//
// Full implementation: Group B Task #1e.
// Mirrors spiral-rs poly.rs multiply, add, scalar_multiply.

#include "params.hpp"
#include "arith.cuh"

#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

// NTT-domain matrix multiply: d_out[m×p] = d_a[m×n] * d_b[n×p] (pointwise per coeff).
void launch_poly_mat_mul(uint64_t* /*d_out*/,
                         const uint64_t* /*d_a*/, uint32_t /*m*/, uint32_t /*n_dim*/,
                         const uint64_t* /*d_b*/, uint32_t /*p*/,
                         cudaStream_t /*stream*/) {
    throw std::runtime_error("poly_mat_mul not yet implemented (Group B Task #1e)");
}

// In-place add: d_result += d_a (both rows×cols matrices in NTT domain).
void launch_poly_add_inplace(uint64_t* /*d_result*/, const uint64_t* /*d_a*/,
                             uint32_t /*rows*/, uint32_t /*cols*/,
                             cudaStream_t /*stream*/) {
    throw std::runtime_error("poly_add_inplace not yet implemented (Group B Task #1e)");
}

// Scalar multiply: d_out = scalar_poly × d_in (both 1×1 poly matrices).
void launch_poly_scalar_mul(uint64_t* /*d_out*/, const uint64_t* /*d_in*/,
                            const uint64_t* /*d_scalar*/,
                            cudaStream_t /*stream*/) {
    throw std::runtime_error("poly_scalar_mul not yet implemented (Group B Task #1e)");
}
