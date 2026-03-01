// NTT-domain polynomial matrix operations for Spiral PIR.
//
// Mirrors spiral-rs poly.rs multiply_poly_acum, add_into_no_reduce, etc.
// All operations work on polynomials already in NTT domain (element-wise per coeff).
// Layout: modulus-major, poly[row,col] at (row*cols+col)*CRT_COUNT*POLY_LEN.

#include "poly_ops.cuh"
#include "params.hpp"
#include "arith.cuh"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

// ── NTT-domain matrix multiply ────────────────────────────────────────────────
//
// Computes d_out[m × p] = d_a[m × n_dim] * d_b[n_dim × p] in NTT domain.
// For each output cell (r, c) and each (crt, z):
//   d_out[r,c,crt,z] = sum_{k=0}^{n_dim-1} barrett_mul(d_a[r,k,crt,z], d_b[k,c,crt,z])
//                                           mod modulus[crt]
//
// Accumulate as u128 to avoid overflow, then Barrett-reduce at end.
// (spiral-rs server.rs uses u128 accumulation in multiply_reg_by_database.)

__global__ void poly_mat_mul_kernel(
    uint64_t* __restrict__ d_out,       // m × p
    const uint64_t* __restrict__ d_a,   // m × n_dim
    const uint64_t* __restrict__ d_b,   // n_dim × p
    uint32_t m, uint32_t n_dim, uint32_t p)
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    // blockIdx.x encodes (out_row * p + out_col) * CRT + crt
    const uint32_t block_id = blockIdx.x;
    const uint32_t total    = m * p * CRT;
    if (block_id >= total) return;

    const uint32_t crt     = block_id % CRT;
    const uint32_t cell    = block_id / CRT;  // out_row * p + out_col
    const uint32_t out_row = cell / p;
    const uint32_t out_col = cell % p;

    const uint64_t modulus = (crt == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;

    // Stride loop: each thread handles multiple coefficients (blockDim.x <= 1024 < N=2048)
    for (uint32_t z = threadIdx.x; z < N; z += blockDim.x) {
        // Accumulate sum in u128 over n_dim terms
        unsigned __int128 acc = 0;
        for (uint32_t k = 0; k < n_dim; ++k) {
            const uint32_t a_idx = (out_row * n_dim + k) * CRT + crt;
            const uint32_t b_idx = (k * p + out_col) * CRT + crt;
            const uint64_t av = d_a[a_idx * N + z];
            const uint64_t bv = d_b[b_idx * N + z];
            acc += static_cast<unsigned __int128>(av) * bv;
        }

        uint64_t result = static_cast<uint64_t>(acc % modulus);
        const uint32_t out_idx = (out_row * p + out_col) * CRT + crt;
        d_out[out_idx * N + z] = result;
    }
}

void launch_poly_mat_mul(uint64_t* d_out,
                          const uint64_t* d_a, uint32_t m, uint32_t n_dim,
                          const uint64_t* d_b, uint32_t p,
                          cudaStream_t stream)
{
    if (m == 0 || n_dim == 0 || p == 0) return;
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;
    const uint32_t num_blocks = m * p * CRT;
    poly_mat_mul_kernel<<<num_blocks, std::min(N, 1024u), 0, stream>>>(d_out, d_a, d_b, m, n_dim, p);
}

// ── In-place add (NTT domain) ─────────────────────────────────────────────────
//
// d_result[r,c,crt,z] += d_a[r,c,crt,z]  (mod modulus[crt])

__global__ void poly_add_inplace_kernel(
    uint64_t* __restrict__ d_result,
    const uint64_t* __restrict__ d_a,
    uint32_t rows, uint32_t cols)
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = rows * cols * CRT * N;
    if (idx >= total) return;

    // Determine CRT modulus for this element
    const uint32_t within_poly = idx % N;
    const uint32_t poly_id     = idx / N;
    const uint32_t crt         = poly_id % CRT;
    const uint64_t modulus     = (crt == 0) ? SpiralParams::MODULUS_0
                                            : SpiralParams::MODULUS_1;

    uint64_t r = d_result[idx] + d_a[idx];
    if (r >= modulus) r -= modulus;
    d_result[idx] = r;

    (void)within_poly;  // suppress unused warning
}

void launch_poly_add_inplace(uint64_t* d_result, const uint64_t* d_a,
                             uint32_t rows, uint32_t cols,
                             cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;
    const uint32_t total = rows * cols * CRT * N;
    const uint32_t threads = 256;
    const uint32_t blocks  = (total + threads - 1) / threads;
    poly_add_inplace_kernel<<<blocks, threads, 0, stream>>>(d_result, d_a, rows, cols);
}

// ── Scalar multiply (NTT domain) ──────────────────────────────────────────────
//
// d_out[r,c,crt,z] = (d_scalar[crt][z] * d_in[r,c,crt,z]) mod modulus[crt]
// for all rows r, cols c.

__global__ void poly_scalar_mul_kernel(
    uint64_t* __restrict__ d_out,
    const uint64_t* __restrict__ d_in,
    const uint64_t* __restrict__ d_scalar,  // 1×1 matrix = CRT * POLY_LEN scalar poly
    uint32_t rows, uint32_t cols)
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = rows * cols * CRT * N;
    if (idx >= total) return;

    const uint32_t z       = idx % N;
    const uint32_t poly_id = idx / N;
    const uint32_t crt     = poly_id % CRT;
    const uint64_t modulus = (crt == 0) ? SpiralParams::MODULUS_0
                                        : SpiralParams::MODULUS_1;

    const uint64_t sv = d_scalar[crt * N + z];
    const uint64_t iv = d_in[idx];

    const uint64_t result = static_cast<uint64_t>(
        (static_cast<unsigned __int128>(sv) * iv) % modulus);
    d_out[idx] = result;
}

void launch_poly_scalar_mul(uint64_t* d_out, const uint64_t* d_in,
                             const uint64_t* d_scalar,
                             uint32_t rows, uint32_t cols,
                             cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;
    const uint32_t total   = rows * cols * CRT * N;
    const uint32_t threads = 256;
    const uint32_t blocks  = (total + threads - 1) / threads;
    poly_scalar_mul_kernel<<<blocks, threads, 0, stream>>>(d_out, d_in, d_scalar, rows, cols);
}

// ── In-place coefficient negate (coefficient domain) ─────────────────────────
//
// For each coefficient: if val != 0, val = qi - val.
// Used by compute_v_folding_neg to avoid a CPU round-trip.

__global__ void poly_negate_kernel(uint64_t* __restrict__ d_data, uint32_t total)
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const uint32_t crt     = (idx / N) % CRT;
    const uint64_t modulus = (crt == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;
    const uint64_t val     = d_data[idx];
    d_data[idx] = (val == 0) ? 0 : modulus - val;
}

void launch_poly_negate(uint64_t* d_data, uint32_t rows, uint32_t cols,
                        cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;
    const uint32_t total   = rows * cols * CRT * N;
    const uint32_t threads = 256;
    const uint32_t blocks  = (total + threads - 1) / threads;
    poly_negate_kernel<<<blocks, threads, 0, stream>>>(d_data, total);
}

// ── Accumulate matrix multiply: d_out += d_a * d_b ───────────────────────────
//
// Used in the inner loop of the database multiply and key-switching accumulation.

__global__ void poly_mat_mul_acum_kernel(
    uint64_t* __restrict__ d_out,       // m × p  (accumulated in place)
    const uint64_t* __restrict__ d_a,   // m × n_dim
    const uint64_t* __restrict__ d_b,   // n_dim × p
    uint32_t m, uint32_t n_dim, uint32_t p)
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t block_id = blockIdx.x;
    if (block_id >= m * p * CRT) return;

    const uint32_t crt     = block_id % CRT;
    const uint32_t cell    = block_id / CRT;
    const uint32_t out_row = cell / p;
    const uint32_t out_col = cell % p;
    const uint64_t modulus = (crt == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;

    // Stride loop: each thread handles multiple coefficients (blockDim.x <= 1024 < N=2048)
    for (uint32_t z = threadIdx.x; z < N; z += blockDim.x) {
        unsigned __int128 acc = static_cast<unsigned __int128>(
            d_out[(out_row * p + out_col) * CRT * N + crt * N + z]);

        for (uint32_t k = 0; k < n_dim; ++k) {
            const uint64_t av = d_a[(out_row * n_dim + k) * CRT * N + crt * N + z];
            const uint64_t bv = d_b[(k * p + out_col) * CRT * N + crt * N + z];
            acc += static_cast<unsigned __int128>(av) * bv;
        }

        d_out[(out_row * p + out_col) * CRT * N + crt * N + z] =
            static_cast<uint64_t>(acc % modulus);
    }
}

void launch_poly_mat_mul_acum(uint64_t* d_out,
                               const uint64_t* d_a, uint32_t m, uint32_t n_dim,
                               const uint64_t* d_b, uint32_t p,
                               cudaStream_t stream)
{
    if (m == 0 || n_dim == 0 || p == 0) return;
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;
    const uint32_t num_blocks = m * p * CRT;
    poly_mat_mul_acum_kernel<<<num_blocks, std::min(N, 1024u), 0, stream>>>(d_out, d_a, d_b, m, n_dim, p);
}
