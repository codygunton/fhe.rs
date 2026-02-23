// Top-level query processing pipeline — STUB
//
// Full implementation: Group E Task #4.
// Mirrors spiral-rs server.rs process_query() exactly.

#include "types.hpp"
#include "params.hpp"
#include "serialization.hpp"

#include <stdexcept>
#include <vector>
#include <cstdint>

// Process one Spiral PIR query end-to-end on GPU.
// Returns response bytes identical to spiral-rs process_query().
std::vector<uint8_t> process_query_gpu(
    const SpiralParams& /*p*/,
    const PublicParamsGPU& /*pp*/,
    const uint8_t* /*query_bytes_raw*/,
    size_t /*query_len*/,
    const DeviceDB& /*db*/,
    cudaStream_t /*stream*/)
{
    // TODO (Group E Task #4):
    //   1. parse_query()
    //   2. expand_query_gpu()
    //   3. For each instance/trial: db_multiply_gpu() → fold_ciphertexts_gpu()
    //   4. pack_gpu() for each instance
    //   5. encode_response()
    throw std::runtime_error("process_query_gpu not yet implemented (Group E Task #4)");
}
