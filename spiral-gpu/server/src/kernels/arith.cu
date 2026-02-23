// Barrett reduction constants and host-side helpers.
// Implements the spiral-rs arith.rs get_barrett_crs algorithm.

#include "arith.cuh"
#include "params.hpp"

#include <cassert>
#include <stdexcept>

// ── Constant memory declarations ──────────────────────────────────────────────
__constant__ BarrettConst d_bc0;   // for MODULUS_0 = 268369921
__constant__ BarrettConst d_bc1;   // for MODULUS_1 = 249561089

// ── Host: compute Barrett cr1 (matches spiral-rs get_barrett_crs) ─────────────
//
// From spiral-rs arith.rs:
//   pub fn get_barrett_crs(modulus: u64) -> (u64, u64) {
//       let two_128 = 1u128 << 128;
//       let cr0 = two_128 / (modulus as u128);   // used for u128 path, not needed here
//       let cr1 = (1u128 << 64) / (modulus as u128);
//       (cr0 as u64, cr1 as u64)
//   }
//
// Only cr1 is used for the u64 Barrett path (barrett_raw_u64).
uint64_t host_barrett_cr1(uint64_t modulus) {
    // (1 << 64) / modulus using 128-bit arithmetic
    unsigned __int128 one64 = static_cast<unsigned __int128>(1) << 64;
    return static_cast<uint64_t>(one64 / modulus);
}

void init_barrett_constants() {
    // Verify our compile-time constants against the formula
    uint64_t cr1_0 = host_barrett_cr1(SpiralParams::MODULUS_0);
    uint64_t cr1_1 = host_barrett_cr1(SpiralParams::MODULUS_1);

    if (cr1_0 != BARRETT_CR1_MOD0) {
        throw std::runtime_error("Barrett cr1 mismatch for MODULUS_0");
    }
    if (cr1_1 != BARRETT_CR1_MOD1) {
        throw std::runtime_error("Barrett cr1 mismatch for MODULUS_1");
    }

    BarrettConst bc0{ SpiralParams::MODULUS_0, cr1_0 };
    BarrettConst bc1{ SpiralParams::MODULUS_1, cr1_1 };

    cudaMemcpyToSymbol(d_bc0, &bc0, sizeof(bc0));
    cudaMemcpyToSymbol(d_bc1, &bc1, sizeof(bc1));
}
