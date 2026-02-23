#pragma once
// Barrett reduction and modular arithmetic for Spiral PIR.
//
// Mirrors spiral-rs src/arith.rs.
// All __device__ functions use precomputed Barrett constants loaded into
// __constant__ memory at startup (see arith.cu).

#include <cstdint>
#include <cuda_runtime.h>

// ── Barrett constants ─────────────────────────────────────────────────────────
// Precomputed for a single modulus.  Stored in __constant__ memory.
struct BarrettConst {
    uint64_t modulus;
    uint64_t cr1;   // only cr1 is needed for u64 × u64 → u64 Barrett
};

// Compile-time constants for the two Spiral moduli.
// Verified against spiral-rs arith.rs get_barrett_crs tests:
//   get_barrett_crs(268369921) → cr1 = 68736257792
//   get_barrett_crs(249561089) → cr1 = 73916747789
//
// Note: these are the "cr1" values from spiral-rs, but the meaning differs from
// the standard Barrett parameterization.  See arith.cu for the exact formula.
static constexpr uint64_t BARRETT_CR1_MOD0 = 68736257792ULL;
static constexpr uint64_t BARRETT_CR1_MOD1 = 73916747789ULL;

// Global Barrett constants for both moduli (defined in arith.cu)
extern __constant__ BarrettConst d_bc0;   // for MODULUS_0 = 268369921
extern __constant__ BarrettConst d_bc1;   // for MODULUS_1 = 249561089

// ── Device functions ──────────────────────────────────────────────────────────

// Barrett reduction for a single u64.
// Matches spiral-rs barrett_raw_u64(input, cr1, modulus).
__device__ __forceinline__ uint64_t barrett_reduce_u64(uint64_t x, const BarrettConst& bc) {
    // From spiral-rs arith.rs:
    //   let w = x as u128;
    //   let q = w * cr1;  (actually cr1 is a scaled 64-bit quotient estimate)
    //   q >> 64 gives the quotient approximation
    //   result = x - q * modulus  (then fix up)
    //
    // The exact spiral-rs formula for barrett_raw_u64:
    //   r = (x as u128 * cr1 as u128) >> 64;
    //   r as u64 * modulus gives the subtracted value
    uint64_t q = static_cast<uint64_t>(
        (static_cast<unsigned __int128>(x) * bc.cr1) >> 64
    );
    uint64_t r = x - q * bc.modulus;
    // Conditional correction (r may be in [0, 2*modulus))
    if (r >= bc.modulus) r -= bc.modulus;
    return r;
}

// Modular addition.
__device__ __forceinline__ uint64_t mod_add(uint64_t a, uint64_t b, uint64_t modulus) {
    uint64_t s = a + b;
    if (s >= modulus) s -= modulus;
    return s;
}

// Modular negation: modulus - a  (returns 0 if a == 0).
__device__ __forceinline__ uint64_t mod_neg(uint64_t a, uint64_t modulus) {
    return (a == 0) ? 0 : modulus - a;
}

// Modular multiplication using Barrett reduction.
// Inputs must be < modulus.
__device__ __forceinline__ uint64_t barrett_mul(uint64_t a, uint64_t b, const BarrettConst& bc) {
    // Full 128-bit product, then reduce mod bc.modulus
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * b;
    // Two-step Barrett on 128-bit: reduce by modulus
    // Simple approach: use __uint128_t % (but that's slow; a proper 128-bit Barrett
    // reduction is implemented in arith.cu for hot paths)
    return static_cast<uint64_t>(prod % bc.modulus);
}

// CRT unpack helpers: extract mod0 / mod1 values from a CRT-packed u64.
// Packed format: (mod0_val & 0xFFFFFFFF) | (mod1_val << 32)
__device__ __forceinline__ uint64_t crt_unpack_lo(uint64_t packed) {
    return packed & 0xFFFFFFFFULL;
}
__device__ __forceinline__ uint64_t crt_unpack_hi(uint64_t packed) {
    return packed >> 32;
}

// ── Host functions ────────────────────────────────────────────────────────────

// Compute Barrett cr1 for a given modulus (matches spiral-rs get_barrett_crs).
// Called once on host to verify the compile-time constants above.
uint64_t host_barrett_cr1(uint64_t modulus);

// Upload Barrett constants for both Spiral moduli to __constant__ memory.
// Must be called before any kernel that uses d_bc0 / d_bc1.
void init_barrett_constants();
