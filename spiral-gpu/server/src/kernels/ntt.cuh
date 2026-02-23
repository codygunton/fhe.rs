#pragma once
// NTT / INTT for Spiral PIR polynomials (poly_len = 2048, two moduli).
//
// Mirrors spiral-rs src/ntt.rs ntt_forward / ntt_inverse.
// Layout: modulus-major (matches spiral-rs PolyMatrixNTT):
//   data[crt * POLY_LEN + coeff]  for a single polynomial
//
// The two moduli use separate twiddle tables stored in __constant__ memory.

#include <cstdint>
#include <cuda_runtime.h>
#include "params.hpp"

// ── Twiddle table API ─────────────────────────────────────────────────────────

// Compute and upload NTT twiddle tables.
// Tables are allocated in device global memory (four arrays × 2048 × 8 bytes = 64 KB;
// fits device memory but would overflow the 64 KB __constant__ limit).
// N^{-1} values are placed in __constant__ memory separately.
// Must be called before any NTT kernel launch.
void init_ntt_tables();

// ── Kernel API ────────────────────────────────────────────────────────────────

// Forward NTT (coefficient domain → NTT domain) for count polynomials.
// d_polys: flat buffer, count * CRT_COUNT * POLY_LEN u64s, modulus-major layout.
// Each polynomial is processed independently (one block per poly per modulus).
void launch_ntt_forward(uint64_t* d_polys, uint32_t count, cudaStream_t stream = 0);

// Inverse NTT (NTT domain → coefficient domain) for count polynomials.
void launch_ntt_inverse(uint64_t* d_polys, uint32_t count, cudaStream_t stream = 0);
