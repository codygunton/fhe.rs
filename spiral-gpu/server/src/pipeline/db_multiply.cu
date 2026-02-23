// First-dimension database multiply — STUB
//
// Full implementation: Group D Task #3c.
// Mirrors spiral-rs server.rs multiply_reg_by_database().

#include "types.hpp"
#include "params.hpp"

#include <stdexcept>
#include <vector>
#include <cstdint>

// Compute: for each (instance, trial), output[num_per] NTT ciphertexts =
// sum over dim0 of v_firstdim[j] * db[instance, trial, :, :, j]
void db_multiply_gpu(DevicePolyMatrix* /*d_out*/,    // [num_per] output ciphertexts
                     const uint64_t* /*d_firstdim*/,  // CRT-packed reoriented query
                     const DeviceDB& /*db*/,
                     uint32_t /*instance*/, uint32_t /*trial*/,
                     const SpiralParams& /*p*/, cudaStream_t /*stream*/) {
    // TODO (Group D Task #3c): implement z-outer parallel dot product kernel.
    throw std::runtime_error("db_multiply_gpu not yet implemented (Group D Task #3c)");
}
