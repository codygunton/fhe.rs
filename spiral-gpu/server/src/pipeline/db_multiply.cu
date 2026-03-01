// First-dimension database multiply for Spiral PIR.
//
// Mirrors spiral-rs server.rs multiply_reg_by_database().
// For each output slot ii ∈ [0, num_per):
//   out[ii][r, 0, crt, z] = sum_{j=0}^{dim0-1} v_firstdim[z, j, r] * db[instance, trial, z, ii, j]
//
// The v_firstdim buffer is in "reoriented" CRT-packed layout:
//   v_firstdim[z * dim0*2 + j*2 + r] = (crt0_val & 0xFFFF) | (crt1_val << 32)
//
// The db buffer is in calc_index order:
//   db[instance*trials*N*num_per*dim0 + trial*N*num_per*dim0 + z*num_per*dim0 + ii*dim0 + j]
//   each word: (ntt_mod0 & 0xFFFF) | (ntt_mod1 << 32)
//
// Output: num_per DevicePolyMatrix, each 2×1, in NTT domain, modulus-major layout.

#include "types.hpp"
#include "params.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

// ── CUDA kernel ───────────────────────────────────────────────────────────────
//
// Parallelizes over (z, ii) pairs.  For each pair, sequentially accumulates
// over dim0 database columns j, computing two dot products (r=0 and r=1) for
// each of the two CRT moduli.
//
// Grid:  ceil((N * num_per) / TPB) blocks
// Block: TPB threads
//
// Each thread handles one (z, ii) pair.
__global__ void db_multiply_kernel(
    uint64_t* __restrict__ d_out,       // num_per × (2×1) NTT matrices, modulus-major
    const uint64_t* __restrict__ d_firstdim,  // CRT-packed reoriented query
    const uint64_t* __restrict__ d_db,  // subset for this (instance, trial), size N*num_per*dim0
    uint32_t dim0,
    uint32_t num_per,
    uint64_t q0,
    uint64_t q1)
{
    constexpr uint32_t N = SpiralParams::POLY_LEN;

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * num_per) return;

    const uint32_t z  = idx / num_per;
    const uint32_t ii = idx % num_per;

    // Accumulators for (crt0, r=0), (crt0, r=1), (crt1, r=0), (crt1, r=1).
    // Use uint64_t and reduce every 128 iterations to stay below 2^63:
    //   each product a*b < q^2 < 2^56, so 128 products sum to < 2^63 < 2^64.
    uint64_t acc_n0_0 = 0, acc_n0_1 = 0;
    uint64_t acc_n1_0 = 0, acc_n1_1 = 0;

    // v_firstdim layout: out[z * dim0*2 + j*2 + r] = (crt0 & 0xFFFF) | (crt1 << 32)
    const uint64_t* fa = d_firstdim + z * (dim0 * 2);

    // db subset (for this instance/trial) layout: [z * num_per * dim0 + ii * dim0 + j]
    const uint64_t* db_z = d_db + static_cast<size_t>(z) * num_per * dim0 + ii * dim0;

    for (uint32_t j = 0; j < dim0; ++j) {
        const uint64_t b_packed = db_z[j];
        const uint32_t b_lo = static_cast<uint32_t>(b_packed);        // mod q0 residue
        const uint32_t b_hi = static_cast<uint32_t>(b_packed >> 32);  // mod q1 residue

        const uint64_t a0_packed = fa[j * 2 + 0];  // row r=0
        const uint64_t a1_packed = fa[j * 2 + 1];  // row r=1

        const uint32_t a0_lo = static_cast<uint32_t>(a0_packed);
        const uint32_t a0_hi = static_cast<uint32_t>(a0_packed >> 32);
        const uint32_t a1_lo = static_cast<uint32_t>(a1_packed);
        const uint32_t a1_hi = static_cast<uint32_t>(a1_packed >> 32);

        acc_n0_0 += static_cast<uint64_t>(a0_lo) * b_lo;
        acc_n0_1 += static_cast<uint64_t>(a1_lo) * b_lo;
        acc_n1_0 += static_cast<uint64_t>(a0_hi) * b_hi;
        acc_n1_1 += static_cast<uint64_t>(a1_hi) * b_hi;

        // Reduce every 128 steps: 128 * q^2 < 128 * 2^56 = 2^63, safe in uint64_t.
        if ((j & 127) == 127) {
            acc_n0_0 %= q0;
            acc_n0_1 %= q0;
            acc_n1_0 %= q1;
            acc_n1_1 %= q1;
        }
    }

    // Final reduction and write output.
    // Output layout for DevicePolyMatrix (2×1, modulus-major):
    //   d_out[ii * 2*CRT*N + row*CRT*N + crt*N + z]
    // Where CRT=2, so:
    //   row=0, crt=0, coeff z: d_out[ii * 2*2*N + 0*2*N + 0*N + z]
    //   row=0, crt=1, coeff z: d_out[ii * 2*2*N + 0*2*N + 1*N + z]
    //   row=1, crt=0, coeff z: d_out[ii * 2*2*N + 1*2*N + 0*N + z]
    //   row=1, crt=1, coeff z: d_out[ii * 2*2*N + 1*2*N + 1*N + z]
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;
    const size_t base = static_cast<size_t>(ii) * 2 * CRT * N;

    d_out[base + 0 * CRT * N + 0 * N + z] = acc_n0_0 % q0;
    d_out[base + 0 * CRT * N + 1 * N + z] = acc_n1_0 % q1;
    d_out[base + 1 * CRT * N + 0 * N + z] = acc_n0_1 % q0;
    d_out[base + 1 * CRT * N + 1 * N + z] = acc_n1_1 % q1;
}

// ── db_multiply_batch_kernel ──────────────────────────────────────────────────
//
// Batched version: processes B queries in a single kernel launch with one
// pass over the database.
//
// Grid:  ceil(B * N * num_per / TPB) blocks
// Block: TPB threads
//
// Each thread handles one (b, z, ii) triple, where b is the query index.
// The DB has no batch dimension — all B queries read the same DB slice.
//
// Output layout (flat buffer, matches launch_ntt_inverse(d_out, B*num_per*2)):
//   Polynomial (b, ii, row) is at offset (b*num_per*2 + ii*2 + row)*CRT*N
//
// d_firstdim_batch layout: query b starts at b * dim0*2*N,
//   within: [z * dim0*2 + j*2 + r] = CRT-packed value for column j, row r, coeff z
__global__ void db_multiply_batch_kernel(
    uint64_t* __restrict__ d_out,                  // [B * num_per * 2 * CRT * N] u64s
    const uint64_t* __restrict__ d_firstdim_batch, // [B * dim0 * 2 * N] u64s
    const uint64_t* __restrict__ d_db,             // [N * num_per * dim0] u64s (this instance/trial)
    uint32_t dim0,
    uint32_t num_per,
    uint32_t batch_size,
    uint64_t q0,
    uint64_t q1)
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * N * num_per) return;

    const uint32_t b   = idx / (N * num_per);
    const uint32_t rem = idx % (N * num_per);
    const uint32_t z   = rem / num_per;
    const uint32_t ii  = rem % num_per;

    uint64_t acc_n0_0 = 0, acc_n0_1 = 0;
    uint64_t acc_n1_0 = 0, acc_n1_1 = 0;

    // Query b's firstdim row z: starts at b*N*dim0*2 + z*dim0*2
    const uint64_t* fa = d_firstdim_batch + static_cast<size_t>(b) * N * dim0 * 2
                         + static_cast<size_t>(z) * (dim0 * 2);

    // DB slice for this (instance, trial): [z * num_per * dim0 + ii * dim0 + j]
    const uint64_t* db_z = d_db + static_cast<size_t>(z) * num_per * dim0 + ii * dim0;

    for (uint32_t j = 0; j < dim0; ++j) {
        const uint64_t b_packed = db_z[j];
        const uint32_t b_lo = static_cast<uint32_t>(b_packed);
        const uint32_t b_hi = static_cast<uint32_t>(b_packed >> 32);

        const uint64_t a0_packed = fa[j * 2 + 0];  // row r=0
        const uint64_t a1_packed = fa[j * 2 + 1];  // row r=1

        const uint32_t a0_lo = static_cast<uint32_t>(a0_packed);
        const uint32_t a0_hi = static_cast<uint32_t>(a0_packed >> 32);
        const uint32_t a1_lo = static_cast<uint32_t>(a1_packed);
        const uint32_t a1_hi = static_cast<uint32_t>(a1_packed >> 32);

        acc_n0_0 += static_cast<uint64_t>(a0_lo) * b_lo;
        acc_n0_1 += static_cast<uint64_t>(a1_lo) * b_lo;
        acc_n1_0 += static_cast<uint64_t>(a0_hi) * b_hi;
        acc_n1_1 += static_cast<uint64_t>(a1_hi) * b_hi;

        // Reduce every 128 steps to avoid overflow
        if ((j & 127) == 127) {
            acc_n0_0 %= q0;
            acc_n0_1 %= q0;
            acc_n1_0 %= q1;
            acc_n1_1 %= q1;
        }
    }

    // Output: polynomial (b, ii, row) at (b*num_per*2 + ii*2 + row)*CRT*N
    const size_t base = (static_cast<size_t>(b) * num_per * 2 + static_cast<size_t>(ii) * 2)
                        * CRT * N;

    d_out[base + 0 * CRT * N + 0 * N + z] = acc_n0_0 % q0;  // row=0, crt=0
    d_out[base + 0 * CRT * N + 1 * N + z] = acc_n1_0 % q1;  // row=0, crt=1
    d_out[base + 1 * CRT * N + 0 * N + z] = acc_n0_1 % q0;  // row=1, crt=0
    d_out[base + 1 * CRT * N + 1 * N + z] = acc_n1_1 % q1;  // row=1, crt=1
}

// ── db_multiply_batch_gpu ─────────────────────────────────────────────────────

void db_multiply_batch_gpu(
    uint64_t* d_batch_out,
    const uint64_t* d_firstdim_batch,
    const DeviceDB& db,
    uint32_t batch_size,
    uint32_t instance, uint32_t trial,
    const SpiralParams& p, cudaStream_t stream)
{
    const uint32_t N       = SpiralParams::POLY_LEN;
    const uint32_t trials  = SpiralParams::N * SpiralParams::N;
    const uint32_t dim0    = p.dim0();
    const uint32_t num_per = p.num_per();

    // Pointer to the db slice for this (instance, trial)
    const size_t db_slice_off = (static_cast<size_t>(instance) * trials + trial)
                                 * N * num_per * dim0;
    const uint64_t* d_db_slice = db.d_data + db_slice_off;

    constexpr uint32_t TPB = 256;
    const uint32_t total   = batch_size * N * num_per;
    const uint32_t blocks  = (total + TPB - 1) / TPB;

    db_multiply_batch_kernel<<<blocks, TPB, 0, stream>>>(
        d_batch_out, d_firstdim_batch, d_db_slice,
        dim0, num_per, batch_size,
        SpiralParams::MODULUS_0, SpiralParams::MODULUS_1);
}

// ── db_multiply_gpu ───────────────────────────────────────────────────────────

void db_multiply_gpu(DevicePolyMatrix* d_out,
                     const uint64_t* d_firstdim,
                     const DeviceDB& db,
                     uint32_t instance, uint32_t trial,
                     const SpiralParams& p, cudaStream_t stream)
{
    const uint32_t N       = SpiralParams::POLY_LEN;
    const uint32_t trials  = SpiralParams::N * SpiralParams::N;
    const uint32_t dim0    = p.dim0();
    const uint32_t num_per = p.num_per();

    // Pointer to the db slice for this (instance, trial)
    const size_t db_slice_off = (static_cast<size_t>(instance) * trials + trial)
                                 * N * num_per * dim0;
    const uint64_t* d_db_slice = db.d_data + db_slice_off;

    // Collect output pointers — d_out[ii] is a 2×1 DevicePolyMatrix
    // All num_per output matrices are already allocated by the caller.

    // We need a contiguous output buffer.  Use a temporary device allocation
    // (num_per * 2*CRT*N u64s) and copy into each d_out[ii].
    //
    // Alternatively, if d_out[ii] are already laid out consecutively we can
    // write directly.  We use a single cudaMalloc to stay simple.
    const size_t out_words = static_cast<size_t>(num_per) * 2 * SpiralParams::CRT_COUNT * N;
    uint64_t* d_tmp = nullptr;
    cudaError_t err = cudaMallocAsync(&d_tmp, out_words * sizeof(uint64_t), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("db_multiply_gpu: cudaMallocAsync failed: ") +
                                 cudaGetErrorString(err));
    }

    constexpr uint32_t TPB = 256;
    const uint32_t total   = N * num_per;
    const uint32_t blocks  = (total + TPB - 1) / TPB;

    db_multiply_kernel<<<blocks, TPB, 0, stream>>>(
        d_tmp, d_firstdim, d_db_slice,
        dim0, num_per,
        SpiralParams::MODULUS_0, SpiralParams::MODULUS_1);

    // Copy each ii slice into d_out[ii]
    const size_t words_per_ct = static_cast<size_t>(2) * SpiralParams::CRT_COUNT * N;
    for (uint32_t ii = 0; ii < num_per; ++ii) {
        cudaMemcpyAsync(d_out[ii].d_data,
                        d_tmp + ii * words_per_ct,
                        words_per_ct * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaFreeAsync(d_tmp, stream);
}
