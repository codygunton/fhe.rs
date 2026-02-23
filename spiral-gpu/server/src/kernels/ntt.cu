// NTT / INTT kernels for Spiral PIR — STUB
//
// Full implementation: Group B Task #1b.
// The NTT uses the same primitive-root twiddle factors as spiral-rs ntt.rs,
// with modulus-major memory layout and the scaled-root-powers butterfly.

#include "ntt.cuh"
#include "arith.cuh"
#include "params.hpp"

#include <stdexcept>

// Twiddle tables stored in device global memory (not constant — four arrays of 2048
// uint64_t = 64 KB total, which would exceed the 64 KB __constant__ limit).
// Allocated and populated by init_ntt_tables().
__device__ uint64_t* d_ntt_fwd_0 = nullptr;   // forward, mod MODULUS_0
__device__ uint64_t* d_ntt_fwd_1 = nullptr;   // forward, mod MODULUS_1
__device__ uint64_t* d_ntt_inv_0 = nullptr;   // inverse, mod MODULUS_0
__device__ uint64_t* d_ntt_inv_1 = nullptr;   // inverse, mod MODULUS_1
// N^{-1} values fit in constant memory
__constant__ uint64_t d_ntt_n_inv_0;
__constant__ uint64_t d_ntt_n_inv_1;

// TODO (Group B Task #1b): implement host-side twiddle computation matching
// spiral-rs ntt.rs calc_psi_powers(), then upload via cudaMemcpyToSymbol.
void init_ntt_tables() {
    // Placeholder: will be implemented in Group B.
    // Must match spiral-rs ntt.rs ntt_forward table generation.
}

// TODO (Group B Task #1b): Cooley-Tukey butterfly kernel, 11 stages, shared memory.
void launch_ntt_forward(uint64_t* /*d_polys*/, uint32_t /*count*/, cudaStream_t /*stream*/) {
    throw std::runtime_error("NTT forward not yet implemented (Group B Task #1b)");
}

void launch_ntt_inverse(uint64_t* /*d_polys*/, uint32_t /*count*/, cudaStream_t /*stream*/) {
    throw std::runtime_error("NTT inverse not yet implemented (Group B Task #1b)");
}
