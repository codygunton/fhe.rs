#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace mulpir::encoding {

/// Encodes raw tile bytes into BFV plaintexts for PIR operations.
///
/// This class handles the conversion between raw byte data (GIS tiles)
/// and BFV plaintext polynomials suitable for homomorphic operations.
/// The encoding follows the same scheme as fhe.rs to ensure compatibility.
class TileEncoder {
public:
    /// Construct a tile encoder with the given BFV context.
    ///
    /// @param context Shared HEonGPU BFV context.
    /// @param config BFV configuration parameters.
    explicit TileEncoder(
        HEContextPtr context,
        const BFVConfig& config = {}
    );

    ~TileEncoder();

    // Disable copy, allow move
    TileEncoder(const TileEncoder&) = delete;
    TileEncoder& operator=(const TileEncoder&) = delete;
    TileEncoder(TileEncoder&&) noexcept;
    TileEncoder& operator=(TileEncoder&&) noexcept;

    /// Encode a single tile into plaintexts.
    ///
    /// If the tile size exceeds BYTES_PER_PLAINTEXT, multiple plaintexts
    /// are returned.
    ///
    /// @param tile_data Raw tile bytes.
    /// @return Vector of encoded plaintexts (usually 1-2 for 30KB tiles).
    std::vector<Plaintext> encode_tile(std::span<const uint8_t> tile_data);

    /// Encode multiple tiles into plaintexts (GPU-accelerated batch).
    ///
    /// This is more efficient than calling encode_tile repeatedly as it
    /// can leverage GPU parallelism for the encoding operations.
    ///
    /// @param tiles Vector of tile data spans.
    /// @return Flattened vector of all encoded plaintexts.
    std::vector<Plaintext> encode_tiles(
        const std::vector<std::span<const uint8_t>>& tiles
    );

    /// Decode plaintexts back to tile bytes (for testing/validation).
    ///
    /// @param encoded Encoded plaintexts representing a single tile.
    /// @param expected_size Expected output size in bytes.
    /// @return Decoded tile bytes.
    std::vector<uint8_t> decode_tile(
        const std::vector<Plaintext>& encoded,
        size_t expected_size
    );

    /// Get the number of plaintexts needed to encode a tile.
    ///
    /// @param tile_size Size of the tile in bytes.
    /// @return Number of plaintexts required.
    size_t plaintexts_per_tile(size_t tile_size) const;

    /// Get the maximum bytes that fit in a single plaintext.
    size_t bytes_per_plaintext() const;

    /// Get bits used per coefficient.
    size_t bits_per_coefficient() const;

private:
    /// Convert raw bytes to polynomial coefficients.
    ///
    /// This matches the fhe.rs transcode_from_bytes function:
    /// - Each coefficient stores BITS_PER_COEFF bits of data
    /// - Bytes are packed into coefficients LSB-first
    ///
    /// @param bytes Input byte data.
    /// @return Vector of coefficients (length = POLY_DEGREE).
    std::vector<uint64_t> bytes_to_coefficients(std::span<const uint8_t> bytes);

    /// Convert polynomial coefficients back to bytes.
    ///
    /// This matches the fhe.rs transcode_to_bytes function.
    ///
    /// @param coeffs Input coefficients.
    /// @param output_size Expected output size in bytes.
    /// @return Decoded bytes.
    std::vector<uint8_t> coefficients_to_bytes(
        std::span<const uint64_t> coeffs,
        size_t output_size
    );

    /// Encode coefficients into a plaintext and transform to NTT form.
    ///
    /// @param coeffs Coefficient vector (length must be POLY_DEGREE).
    /// @return Encoded plaintext in NTT form.
    Plaintext encode_and_ntt(const std::vector<uint64_t>& coeffs);

    /// Decode a plaintext from NTT form back to coefficients.
    ///
    /// @param pt Plaintext in NTT form.
    /// @return Decoded coefficients.
    std::vector<uint64_t> intt_and_decode(const Plaintext& pt);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    BFVConfig config_;
};

}  // namespace mulpir::encoding
