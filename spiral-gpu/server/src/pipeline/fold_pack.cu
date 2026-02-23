// Second-dimension fold and packing — STUB
//
// Full implementation: Group D Task #3d.
// Mirrors spiral-rs server.rs fold_ciphertexts() and pack().

#include "types.hpp"
#include "params.hpp"

#include <stdexcept>
#include <vector>

// Fold num_per ciphertexts down to 1 using nu_2 GSW keys.
// Modifies v_cts in-place (num_per → num_per/2 → … → 1).
void fold_ciphertexts_gpu(std::vector<DevicePolyMatrix>& /*v_cts*/,
                          const std::vector<DevicePolyMatrix>& /*v_gsw*/,
                          const SpiralParams& /*p*/, cudaStream_t /*stream*/) {
    // TODO (Group D Task #3d): implement fold using gadget decomp + key multiply.
    throw std::runtime_error("fold_ciphertexts_gpu not yet implemented (Group D Task #3d)");
}

// Pack n×n result ciphertexts into one (n+1)×n ciphertext using v_packing.
DevicePolyMatrix pack_gpu(const std::vector<DevicePolyMatrix>& /*v_ct*/,
                          const PublicParamsGPU& /*pp*/,
                          const SpiralParams& /*p*/, cudaStream_t /*stream*/) {
    // TODO (Group D Task #3d): implement pack using v_packing gadget multiply.
    throw std::runtime_error("pack_gpu not yet implemented (Group D Task #3d)");
}
