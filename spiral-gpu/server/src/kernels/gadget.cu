// Gadget decomposition kernel for Spiral PIR.
//
// Mirrors spiral-rs gadget.rs gadget_invert_rdim.
//
// Algorithm (from gadget.rs):
//   num_elems = out.rows / rdim
//   bits_per  = floor(modulus_log2 / num_elems) + 1   (or 1 if modulus_log2 == num_elems)
//   mask      = (1 << bits_per) - 1
//
//   for each column i (0..cols), row j (0..rdim), coefficient z (0..POLY_LEN):
//     for k in 0..num_elems:
//       bit_offs = min(k * bits_per, 64)
//       out[j + k*rdim, i][z] = (inp[j, i][z] >> bit_offs) & mask
//
// The `bits_per` and `num_elems` parameters are passed in (computed by caller
// from the modulus and the requested decomposition dimension).
//
// Layout: modulus-major, CRT_COUNT * POLY_LEN u64s per polynomial cell.
// Matrix layout (row-major): poly[row, col] lives at
//   base + (row * cols + col) * CRT_COUNT * POLY_LEN

#include "params.hpp"
#include "arith.cuh"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

// Kernel: one thread per (output_row, col, crt, coeff) — but we batch over
// multiple (j, k, i, crt, z) tuples.
//
// Grid layout: blockIdx.x = output_row (= j + k*rdim), blockIdx.y = col (i)
//              blockIdx.z = crt modulus index
// threadIdx.x = coefficient index z
//
// For each output polynomial cell (out_row, col, crt):
//   k = out_row / rdim   (limb index)
//   j = out_row % rdim   (input row)
//   Read inp[j, col, crt, z] and extract limb k.
__global__ void gadget_invert_kernel(
    uint64_t* __restrict__ d_out,       // (rdim*num_elems) × cols polynomial matrix
    const uint64_t* __restrict__ d_inp, // rdim × cols polynomial matrix
    uint32_t rdim,
    uint32_t cols,
    uint32_t num_elems,
    uint32_t bits_per)
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t out_row = blockIdx.x;  // 0..rdim*num_elems
    const uint32_t col     = blockIdx.y;  // 0..cols
    const uint32_t crt     = blockIdx.z;  // 0..CRT_COUNT

    if (out_row >= rdim * num_elems || col >= cols || crt >= CRT) return;

    const uint32_t k = out_row / rdim;  // limb index
    const uint32_t j = out_row % rdim;  // input row

    const uint32_t inp_poly_idx = j * cols + col;
    const uint32_t out_poly_idx = out_row * cols + col;

    // Bit offset for limb k
    const uint32_t bit_offs = (k * bits_per < 64) ? (k * bits_per) : 64;
    const uint64_t mask     = bits_per < 64 ? ((1ULL << bits_per) - 1ULL) : ~0ULL;

    // Stride loop: each thread handles multiple coefficients (blockDim.x <= 1024 < N=2048)
    for (uint32_t z = threadIdx.x; z < N; z += blockDim.x) {
        const uint64_t inp_val = d_inp[(inp_poly_idx * CRT + crt) * N + z];
        const uint64_t out_val = (inp_val >> bit_offs) & mask;
        d_out[(out_poly_idx * CRT + crt) * N + z] = out_val;
    }
}

// Launch gadget inversion on the GPU.
//
// d_out: output matrix buffer  — (rdim * num_elems) rows × cols cols
// d_inp: input matrix buffer   — rdim rows × cols cols
// Both buffers contain polynomials in coefficient domain, modulus-major layout.
// num_elems: number of limbs per coefficient (bits_per bits each)
// bits_per:  bits extracted per limb
void launch_gadget_invert(uint64_t* d_out, const uint64_t* d_inp,
                          uint32_t rdim, uint32_t cols,
                          uint32_t num_elems, uint32_t bits_per,
                          cudaStream_t stream)
{
    if (rdim == 0 || cols == 0 || num_elems == 0) return;

    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t out_rows = rdim * num_elems;

    dim3 grid(out_rows, cols, CRT);
    dim3 block(std::min(N, 1024u));  // CUDA max threads/block = 1024; stride loop handles rest

    gadget_invert_kernel<<<grid, block, 0, stream>>>(
        d_out, d_inp, rdim, cols, num_elems, bits_per);
}

// ── CRT-composing gadget inversion ────────────────────────────────────────────
//
// Like gadget_invert but first CRT-composes the two slots back into the full
// combined modulus value (q0*q1 ≈ 2^56), then extracts bits from that value.
// The same small limb value is written to BOTH CRT output slots.
//
// Required for coefficient_expansion and fold_ciphertexts where the INTT output
// values can span the full 56-bit range and gadget decomposition must be done
// on the reconstructed integer, not the per-slot residues.
//
// CRT composition: full = (a0 * q1 * inv(q1,q0) + a1 * q0 * inv(q0,q1)) mod (q0*q1)
// Precomputed: mod0_inv_mod1 = q0^{-1} mod q1,  mod1_inv_mod0 = q1^{-1} mod q0
// Then: full = a0 + q0 * ((a1 - a0) * mod0_inv_mod1 mod q1)
__global__ void gadget_invert_crt_kernel(
    uint64_t* __restrict__ d_out,
    const uint64_t* __restrict__ d_inp,
    uint32_t rdim,
    uint32_t cols,
    uint32_t num_elems,
    uint32_t bits_per,
    uint64_t q0,
    uint64_t q1,
    uint64_t mod0_inv_mod1)  // q0^{-1} mod q1
{
    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT = SpiralParams::CRT_COUNT;

    const uint32_t out_row = blockIdx.x;  // 0..rdim*num_elems
    const uint32_t col     = blockIdx.y;  // 0..cols

    if (out_row >= rdim * num_elems || col >= cols) return;

    const uint32_t k = out_row / rdim;
    const uint32_t j = out_row % rdim;

    const uint32_t inp_poly_idx = j * cols + col;
    const uint32_t out_poly_idx = out_row * cols + col;
    const uint32_t bit_offs = k * bits_per;

    // Stride loop: each thread handles multiple coefficients (blockDim.x <= 1024 < N=2048)
    for (uint32_t z = threadIdx.x; z < N; z += blockDim.x) {
        // Read both CRT slots
        const uint64_t a0 = d_inp[(inp_poly_idx * CRT + 0) * N + z];  // val mod q0
        const uint64_t a1 = d_inp[(inp_poly_idx * CRT + 1) * N + z];  // val mod q1

        // CRT-compose: full = a0 + q0 * ((a1 - a0 + q1) * mod0_inv_mod1 mod q1)
        uint64_t diff = (a1 + q1 - a0 % q1) % q1;
        uint64_t carry = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(diff) * mod0_inv_mod1) % q1);
        unsigned __int128 full_val = static_cast<unsigned __int128>(a0) +
                                      static_cast<unsigned __int128>(q0) * carry;

        // Extract bits_per-bit limb k from the full value
        uint64_t out_val = static_cast<uint64_t>(full_val >> bit_offs);
        if (bits_per < 64) {
            out_val &= (1ULL << bits_per) - 1ULL;
        }

        // Write same small value to BOTH CRT output slots
        d_out[(out_poly_idx * CRT + 0) * N + z] = out_val;
        d_out[(out_poly_idx * CRT + 1) * N + z] = out_val;
    }
}

void launch_gadget_invert_crt(uint64_t* d_out, const uint64_t* d_inp,
                               uint32_t rdim, uint32_t cols,
                               uint32_t num_elems, uint32_t bits_per,
                               cudaStream_t stream)
{
    if (rdim == 0 || cols == 0 || num_elems == 0) return;

    constexpr uint32_t N   = SpiralParams::POLY_LEN;
    constexpr uint64_t q0  = SpiralParams::MODULUS_0;
    constexpr uint64_t q1  = SpiralParams::MODULUS_1;

    // Compute q0^{-1} mod q1 once at startup via extended Euclidean.
    // q0 = 268369921, q1 = 249561089
    static const uint64_t MOD0_INV_MOD1 = []() -> uint64_t {
        int64_t t = 0, newt = 1;
        int64_t r = (int64_t)q1, newr = (int64_t)q0;
        while (newr != 0) {
            int64_t q = r / newr, tmp;
            tmp = t - q * newt; t = newt; newt = tmp;
            tmp = r - q * newr; r = newr; newr = tmp;
        }
        if (t < 0) t += (int64_t)q1;
        return static_cast<uint64_t>(t);
    }();

    const uint32_t out_rows = rdim * num_elems;

    dim3 grid(out_rows, cols);
    dim3 block(std::min(N, 1024u));  // CUDA max threads/block = 1024; stride loop handles rest

    gadget_invert_crt_kernel<<<grid, block, 0, stream>>>(
        d_out, d_inp, rdim, cols, num_elems, bits_per,
        q0, q1, MOD0_INV_MOD1);
}
