// Spiral binary format: parse PublicParameters + Query, encode response.
//
// Full implementation: Group C Task #2.
// Mirrors spiral-rs client.rs PublicParameters::deserialize() and Query::deserialize(),
// plus server.rs encode().

#include "serialization.hpp"
#include "params.hpp"
#include "types.hpp"

#include <stdexcept>

// ── parse_public_params ───────────────────────────────────────────────────────
PublicParamsGPU parse_public_params(const uint8_t* /*data*/, size_t len,
                                    const SpiralParams& p) {
    if (len != p.setup_bytes()) {
        throw std::runtime_error("parse_public_params: length mismatch: got "
            + std::to_string(len) + ", expected " + std::to_string(p.setup_bytes()));
    }
    // TODO (Group C Task #2): deserialize matrices and upload to device.
    throw std::runtime_error("parse_public_params not yet implemented (Group C Task #2)");
}

// ── parse_query ───────────────────────────────────────────────────────────────
CiphertextGPU parse_query(const uint8_t* /*data*/, size_t len, const SpiralParams& p) {
    if (len != p.query_bytes()) {
        throw std::runtime_error("parse_query: length mismatch: got "
            + std::to_string(len) + ", expected " + std::to_string(p.query_bytes()));
    }
    // TODO (Group C Task #2): deserialize query polynomial and upload to device.
    throw std::runtime_error("parse_query not yet implemented (Group C Task #2)");
}

// ── encode_response ───────────────────────────────────────────────────────────
std::vector<uint8_t> encode_response(const std::vector<DevicePolyMatrix>& /*result_mats*/,
                                     const SpiralParams& p) {
    // TODO (Group C Task #2): INTT on device, download, apply rescale quantization.
    throw std::runtime_error("encode_response not yet implemented (Group C Task #2)");
    (void)p;
}
