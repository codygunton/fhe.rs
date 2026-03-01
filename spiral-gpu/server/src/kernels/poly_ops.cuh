#pragma once
// NTT-domain polynomial matrix operations for Spiral PIR.
//
// All buffers are in NTT domain, modulus-major layout:
//   poly[row, col] at offset (row*cols + col) * CRT_COUNT * POLY_LEN.

#include <cstdint>
#include <cuda_runtime.h>

// Matrix multiply: d_out[m×p] = d_a[m×n] * d_b[n×p]  (NTT domain, pointwise per coeff).
void launch_poly_mat_mul(uint64_t* d_out,
                          const uint64_t* d_a, uint32_t m, uint32_t n_dim,
                          const uint64_t* d_b, uint32_t p,
                          cudaStream_t stream = 0);

// Accumulate: d_out += d_a * d_b  (NTT domain).
void launch_poly_mat_mul_acum(uint64_t* d_out,
                               const uint64_t* d_a, uint32_t m, uint32_t n_dim,
                               const uint64_t* d_b, uint32_t p,
                               cudaStream_t stream = 0);

// In-place add: d_result += d_a  (NTT domain, rows×cols matrices).
void launch_poly_add_inplace(uint64_t* d_result, const uint64_t* d_a,
                             uint32_t rows, uint32_t cols,
                             cudaStream_t stream = 0);

// Scalar multiply: d_out[rows×cols] = d_scalar[1×1] * d_in[rows×cols]  (NTT domain).
void launch_poly_scalar_mul(uint64_t* d_out, const uint64_t* d_in,
                             const uint64_t* d_scalar,
                             uint32_t rows, uint32_t cols,
                             cudaStream_t stream = 0);

// In-place negate all coefficients (coefficient domain): val = (val == 0) ? 0 : qi - val.
// Handles CRT layout automatically (rows×cols matrix).
void launch_poly_negate(uint64_t* d_data, uint32_t rows, uint32_t cols,
                        cudaStream_t stream = 0);
