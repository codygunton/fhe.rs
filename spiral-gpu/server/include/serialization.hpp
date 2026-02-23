#pragma once
// Spiral binary format: parse PublicParameters + Query, encode response.
//
// The serialized formats are defined by spiral-rs:
//   - PublicParameters::serialize() / deserialize()  in src/client.rs
//   - Query::serialize() / deserialize()             in src/client.rs
//   - encode()                                       in src/server.rs
//
// All u64s are in native-endian (little-endian on x86).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "params.hpp"
#include "types.hpp"

// Parse raw setup bytes from POST /api/setup into GPU-resident PublicParamsGPU.
// Uploads all polynomial matrices to device memory.
// Throws std::runtime_error on length mismatch or CUDA error.
PublicParamsGPU parse_public_params(const uint8_t* data, size_t len,
                                    const SpiralParams& p);

// Parse raw query bytes from POST /api/private-read into a CiphertextGPU.
// The query is a single polynomial (2×1 matrix, rows-1=1 row serialized).
// Throws std::runtime_error on length mismatch.
CiphertextGPU parse_query(const uint8_t* data, size_t len, const SpiralParams& p);

// Encode GPU result matrices into the spiral-rs response byte format.
// result_mats is a vector of (N+1)×N packed ciphertexts, one per instance.
// Matches spiral-rs server.rs encode() exactly.
std::vector<uint8_t> encode_response(const std::vector<DevicePolyMatrix>& result_mats,
                                     const SpiralParams& p);
