#pragma once
// Galois automorphism for Spiral PIR polynomials.
//
// Mirrors spiral-rs poly.rs automorph_poly: sends X → X^t in the cyclotomic ring
// Z[X]/(X^N + 1), operating in coefficient domain.

#include <cstdint>
#include <cuda_runtime.h>

// Apply the Galois automorphism X → X^t to a batch of polynomials.
//
// d_out: output buffer (modulus-major, num_polys * CRT_COUNT * POLY_LEN u64s)
//        Must be distinct from d_in.  Must be zeroed before this call.
// d_in:  input buffer (same layout as d_out)
// t:     Galois element (must be odd, in range [1, 2*POLY_LEN))
// num_polys: number of polynomials to process
// stream:    CUDA stream
void launch_automorph(uint64_t* d_out, const uint64_t* d_in,
                      uint32_t t, uint32_t num_polys,
                      cudaStream_t stream = 0);
