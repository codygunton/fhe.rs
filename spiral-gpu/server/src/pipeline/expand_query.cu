// Query coefficient expansion — STUB
//
// Full implementation: Group D Task #3b.
// Mirrors spiral-rs server.rs coefficient_expansion() and regev_to_gsw().

#include "types.hpp"
#include "params.hpp"

#include <stdexcept>
#include <vector>

// Returns (v_reg_reoriented, v_folding):
//   v_reg_reoriented: dim0 ciphertexts in CRT-packed reoriented layout
//   v_folding:        nu_2 GSW ciphertexts for second-dimension folding
std::pair<std::vector<uint64_t>, std::vector<DevicePolyMatrix>>
expand_query_gpu(const CiphertextGPU& /*ct*/, const PublicParamsGPU& /*pp*/,
                 const SpiralParams& /*p*/, cudaStream_t /*stream*/) {
    // TODO (Group D Task #3b): implement coefficient_expansion tree on GPU.
    throw std::runtime_error("expand_query_gpu not yet implemented (Group D Task #3b)");
}
