// Galois automorphism kernel for Spiral PIR.
//
// Mirrors spiral-rs poly.rs automorph_poly / automorph:
//   out[i] = (-1)^sign * in[mapped_index]
// where the permutation sends coefficient i to the position corresponding to
// X^i → X^(i*t mod 2N) in the cyclotomic ring Z[X]/(X^N + 1).
//
// In spiral-rs, automorph(i, t) is defined as:
//   new_i = (i * t) % (2 * poly_len)
//   if new_i >= poly_len: out[new_i - poly_len] -= in[i]  (subtract, i.e. negate)
//   else:                 out[new_i]             += in[i]  (add)
//
// Since we're working in coefficient domain with modular arithmetic, negate =
// (modulus - val) mod modulus.
//
// Layout: modulus-major, CRT_COUNT * POLY_LEN u64s per polynomial.
// The function accepts a batch of `num_polys` polynomials.

#include "params.hpp"
#include "arith.cuh"

#include <cstdint>
#include <cuda_runtime.h>

// One block per (polynomial, CRT modulus) pair.
// Threads handle individual coefficients.
__global__ void automorph_kernel(
    uint64_t* __restrict__ d_out,
    const uint64_t* __restrict__ d_in,
    uint32_t t,          // Galois element (must be odd)
    uint32_t num_polys)
{
    constexpr uint32_t N     = SpiralParams::POLY_LEN;
    constexpr uint32_t TWO_N = 2 * N;

    const uint32_t block_id = blockIdx.x;
    if (block_id >= num_polys * SpiralParams::CRT_COUNT) return;

    const uint32_t poly_idx = block_id / SpiralParams::CRT_COUNT;
    const uint32_t crt_idx  = block_id % SpiralParams::CRT_COUNT;
    const uint64_t modulus  = (crt_idx == 0) ? SpiralParams::MODULUS_0
                                              : SpiralParams::MODULUS_1;

    const uint32_t base = (poly_idx * SpiralParams::CRT_COUNT + crt_idx) * N;
    const uint64_t* in  = d_in  + base;
    uint64_t*       out = d_out + base;

    // Each thread handles one coefficient i.
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
        uint32_t new_i = static_cast<uint32_t>(
            (static_cast<uint64_t>(i) * t) % TWO_N);

        uint64_t val = in[i];

        if (new_i >= N) {
            // Coefficient maps to the "negative" half: negate and write to [new_i - N]
            uint64_t neg_val = (val == 0) ? 0 : (modulus - val);
            out[new_i - N] = neg_val;
        } else {
            out[new_i] = val;
        }
    }
}

// Apply Galois automorphism X → X^t to a batch of polynomials in coefficient domain.
// d_out and d_in must be distinct (non-aliased) buffers.
// The output buffer must be zeroed before calling (the scatter may leave holes).
void launch_automorph(uint64_t* d_out, const uint64_t* d_in,
                      uint32_t t, uint32_t num_polys,
                      cudaStream_t stream)
{
    if (num_polys == 0) return;
    const uint32_t num_blocks = num_polys * SpiralParams::CRT_COUNT;
    const uint32_t threads = 256;
    automorph_kernel<<<num_blocks, threads, 0, stream>>>(d_out, d_in, t, num_polys);
}
