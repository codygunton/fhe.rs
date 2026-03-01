#pragma once
// NTT / INTT for Spiral PIR polynomials (poly_len = 2048, two moduli).
//
// Mirrors spiral-rs src/ntt.rs ntt_forward / ntt_inverse (non-AVX2 path).
// Layout: modulus-major (matches spiral-rs PolyMatrixNTT):
//   data[crt * POLY_LEN + coeff]  for a single polynomial
//
// The two moduli use separate twiddle tables stored in __device__ global memory.
// Four tables per modulus:
//   fwd   — forward root powers (bit-reversed)
//   fwdp  — forward prime (scaled: (inp << 32) / modulus as u32)
//   inv   — inverse root powers (bit-reversed, div2 applied)
//   invp  — inverse prime (scaled from inv)

#include <cstdint>
#include <cuda_runtime.h>
#include "params.hpp"

// ── Twiddle table API ─────────────────────────────────────────────────────────

// Compute and upload NTT twiddle tables to device global memory.
// Validates against spiral-rs reference values:
//   tables[0][2][0] == 134184961  (MODULUS_0 inverse table element 0)
//   tables[0][2][1] == 96647580
// Must be called before any NTT kernel launch.
void init_ntt_tables();

// ── Kernel API ────────────────────────────────────────────────────────────────

// Forward NTT (coefficient domain → NTT domain) for count polynomials.
// d_polys: flat buffer, count * CRT_COUNT * POLY_LEN u64s, modulus-major layout.
// Each polynomial is processed independently (one block per poly per modulus).
void launch_ntt_forward(uint64_t* d_polys, uint32_t count, cudaStream_t stream = 0);

// Inverse NTT (NTT domain → coefficient domain) for count polynomials.
void launch_ntt_inverse(uint64_t* d_polys, uint32_t count, cudaStream_t stream = 0);
