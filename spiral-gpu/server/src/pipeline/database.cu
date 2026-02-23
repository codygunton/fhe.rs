// Database loading and GPU upload — STUB
//
// Full implementation: Group D Task #3a.
// Mirrors spiral-rs server.rs load_db_from_seek() exactly.

#include "types.hpp"
#include "params.hpp"

#include <stdexcept>
#include <cstdint>

DeviceDB load_db_to_gpu(const uint8_t* /*raw_tiles*/, size_t /*file_size*/,
                        const SpiralParams& /*p*/) {
    // TODO (Group D Task #3a): read tiles, decompose into 20-bit coefficients,
    // recenter, NTT, CRT-pack, calc_index reorder, cudaMemcpy to device.
    throw std::runtime_error("load_db_to_gpu not yet implemented (Group D Task #3a)");
}
