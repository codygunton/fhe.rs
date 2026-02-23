// Galois automorphism kernel — STUB
//
// Full implementation: Group B Task #1c.
// Mirrors spiral-rs poly.rs automorph_poly: coefficient permutation X → X^t.

#include "params.hpp"
#include "arith.cuh"

#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

// Apply automorphism X → X^t to a batch of polynomials in-place (coefficient domain).
// d_out and d_in must be distinct (non-aliased) buffers.
// num_polys: number of row-major polynomials in the batch.
// Layout: modulus-major, CRT_COUNT * POLY_LEN u64s per polynomial.
void launch_automorph(uint64_t* /*d_out*/, const uint64_t* /*d_in*/,
                      uint32_t /*t*/, uint32_t /*num_polys*/,
                      cudaStream_t /*stream*/) {
    throw std::runtime_error("automorph not yet implemented (Group B Task #1c)");
}
