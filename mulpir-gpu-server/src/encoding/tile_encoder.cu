#include "encoding/tile_encoder.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mulpir::encoding {

struct TileEncoder::Impl {
    HEContextPtr context;
    mulpir::HEEncoder encoder;

    explicit Impl(HEContextPtr ctx)
        : context(ctx)
        , encoder(context) {
    }
};

TileEncoder::TileEncoder(
    HEContextPtr context,
    const BFVConfig& config
)
    : impl_(std::make_unique<Impl>(std::move(context)))
    , config_(config) {
}

TileEncoder::~TileEncoder() = default;

TileEncoder::TileEncoder(TileEncoder&&) noexcept = default;
TileEncoder& TileEncoder::operator=(TileEncoder&&) noexcept = default;

size_t TileEncoder::bytes_per_plaintext() const {
    return config_.BYTES_PER_PLAINTEXT;
}

size_t TileEncoder::bits_per_coefficient() const {
    return config_.BITS_PER_COEFF;
}

size_t TileEncoder::plaintexts_per_tile(size_t tile_size) const {
    return (tile_size + bytes_per_plaintext() - 1) / bytes_per_plaintext();
}

std::vector<uint64_t> TileEncoder::bytes_to_coefficients(std::span<const uint8_t> bytes) {
    // Match fhe.rs transcode_from_bytes algorithm exactly
    const size_t nbits = config_.BITS_PER_COEFF;
    const uint64_t mask = (1ULL << nbits) - 1;

    // Output: POLY_DEGREE coefficients
    const size_t nelements = config_.POLY_DEGREE;
    std::vector<uint64_t> out;
    out.reserve(nelements);

    __uint128_t current_value = 0;
    size_t current_value_nbits = 0;
    size_t current_index = 0;

    while (out.size() < nelements) {
        // Load more bits if needed
        while (current_value_nbits < nbits && current_index < bytes.size()) {
            current_value |= static_cast<__uint128_t>(bytes[current_index]) << current_value_nbits;
            current_value_nbits += 8;
            current_index++;
        }

        // Extract coefficient
        out.push_back(static_cast<uint64_t>(current_value & mask));
        current_value >>= nbits;

        if (current_value_nbits >= nbits) {
            current_value_nbits -= nbits;
        } else {
            current_value_nbits = 0;
        }
    }

    return out;
}

std::vector<uint8_t> TileEncoder::coefficients_to_bytes(
    std::span<const uint64_t> coeffs,
    size_t output_size
) {
    // Match fhe.rs transcode_to_bytes algorithm exactly
    const size_t nbits = config_.BITS_PER_COEFF;
    const uint64_t mask = (1ULL << nbits) - 1;

    std::vector<uint8_t> out;
    out.reserve(output_size);

    __uint128_t current_value = 0;
    size_t current_value_nbits = 0;
    size_t current_index = 0;

    while (out.size() < output_size) {
        // Load more bits from coefficients if needed
        while (current_value_nbits < 8 && current_index < coeffs.size()) {
            const uint64_t coeff = coeffs[current_index] & mask;
            current_value |= static_cast<__uint128_t>(coeff) << current_value_nbits;
            current_value_nbits += nbits;
            current_index++;
        }

        // Extract byte
        if (current_value_nbits >= 8) {
            out.push_back(static_cast<uint8_t>(current_value & 0xFF));
            current_value >>= 8;
            current_value_nbits -= 8;
        } else if (out.size() < output_size) {
            // Pad with zeros if we run out of coefficient bits
            out.push_back(static_cast<uint8_t>(current_value & 0xFF));
            current_value = 0;
            current_value_nbits = 0;
        }
    }

    return out;
}

Plaintext TileEncoder::encode_and_ntt(const std::vector<uint64_t>& coeffs) {
    if (coeffs.size() != config_.POLY_DEGREE) {
        throw std::invalid_argument("Coefficient vector must have POLY_DEGREE elements");
    }

    Plaintext pt(impl_->context);
    impl_->encoder.encode(pt, coeffs);
    return pt;
}

std::vector<uint64_t> TileEncoder::intt_and_decode(const Plaintext& pt) {
    std::vector<uint64_t> coeffs;
    // HEonGPU decode takes non-const reference for GPU memory management
    auto& pt_ref = const_cast<Plaintext&>(pt);
    impl_->encoder.decode(coeffs, pt_ref);
    return coeffs;
}

std::vector<Plaintext> TileEncoder::encode_tile(std::span<const uint8_t> tile_data) {
    const size_t bytes_per_pt = bytes_per_plaintext();
    const size_t num_pts = plaintexts_per_tile(tile_data.size());

    std::vector<Plaintext> result;
    result.reserve(num_pts);

    for (size_t i = 0; i < num_pts; ++i) {
        const size_t start = i * bytes_per_pt;
        const size_t end = std::min(start + bytes_per_pt, tile_data.size());
        const size_t chunk_size = end - start;

        // Get chunk of tile data
        std::span<const uint8_t> chunk = tile_data.subspan(start, chunk_size);

        // Convert bytes to coefficients
        auto coeffs = bytes_to_coefficients(chunk);

        // Encode and transform to NTT
        result.push_back(encode_and_ntt(coeffs));
    }

    return result;
}

std::vector<Plaintext> TileEncoder::encode_tiles(
    const std::vector<std::span<const uint8_t>>& tiles
) {
    std::vector<Plaintext> result;

    // Pre-calculate total plaintexts needed
    size_t total_pts = 0;
    for (const auto& tile : tiles) {
        total_pts += plaintexts_per_tile(tile.size());
    }
    result.reserve(total_pts);

    // Encode each tile
    for (const auto& tile : tiles) {
        auto encoded = encode_tile(tile);
        result.insert(result.end(),
                     std::make_move_iterator(encoded.begin()),
                     std::make_move_iterator(encoded.end()));
    }

    return result;
}

std::vector<uint8_t> TileEncoder::decode_tile(
    const std::vector<Plaintext>& encoded,
    size_t expected_size
) {
    const size_t bytes_per_pt = bytes_per_plaintext();

    std::vector<uint8_t> result;
    result.reserve(expected_size);

    for (size_t i = 0; i < encoded.size(); ++i) {
        // Decode from NTT and get coefficients
        auto coeffs = intt_and_decode(encoded[i]);

        // Calculate how many bytes this plaintext contributes
        const size_t offset = i * bytes_per_pt;
        const size_t remaining = expected_size - offset;
        const size_t chunk_size = std::min(bytes_per_pt, remaining);

        // Convert coefficients to bytes
        auto bytes = coefficients_to_bytes(coeffs, chunk_size);

        result.insert(result.end(), bytes.begin(), bytes.end());
    }

    // Truncate to expected size
    result.resize(expected_size);
    return result;
}

}  // namespace mulpir::encoding
