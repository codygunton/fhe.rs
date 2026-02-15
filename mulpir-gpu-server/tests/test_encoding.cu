#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "config.hpp"
#include "encoding/tile_encoder.hpp"
#include "test_helpers.hpp"

using namespace mulpir;
using namespace mulpir::encoding;

class TileEncoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        context_ = test::create_test_context();
        encoder_ = std::make_unique<TileEncoder>(context_);
    }

    HEContextPtr context_;
    std::unique_ptr<TileEncoder> encoder_;
    std::mt19937_64 rng_{12345};
};

TEST_F(TileEncoderTest, BytesPerPlaintext) {
    // Expected: BITS_PER_COEFF * POLY_DEGREE / 8 = 20 * 8192 / 8 = 20480
    EXPECT_EQ(encoder_->bytes_per_plaintext(), 20480);
}

TEST_F(TileEncoderTest, BitsPerCoefficient) {
    // Expected: floor(log2(0x1E0001)) = 20
    EXPECT_EQ(encoder_->bits_per_coefficient(), 20);
}

TEST_F(TileEncoderTest, PlaintextsPerSmallTile) {
    // 1KB tile should fit in 1 plaintext
    EXPECT_EQ(encoder_->plaintexts_per_tile(1024), 1);
}

TEST_F(TileEncoderTest, PlaintextsPerLargeTile) {
    // 30KB tile needs 2 plaintexts (30720 / 20480 = 1.5, rounded up)
    EXPECT_EQ(encoder_->plaintexts_per_tile(30720), 2);
}

TEST_F(TileEncoderTest, PlaintextsPerMaxTile) {
    // Max tile size that fits in 1 plaintext
    EXPECT_EQ(encoder_->plaintexts_per_tile(20480), 1);
    EXPECT_EQ(encoder_->plaintexts_per_tile(20481), 2);
}

TEST_F(TileEncoderTest, EncodeSmallTile) {
    // Create a small tile
    std::vector<uint8_t> tile(1024);
    for (size_t i = 0; i < tile.size(); ++i) {
        tile[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Encode
    auto encoded = encoder_->encode_tile({tile.data(), tile.size()});

    // Should produce 1 plaintext
    EXPECT_EQ(encoded.size(), 1);
}

TEST_F(TileEncoderTest, EncodeLargeTile) {
    // Create a 30KB tile
    std::vector<uint8_t> tile(30720);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (auto& b : tile) {
        b = dist(rng_);
    }

    // Encode
    auto encoded = encoder_->encode_tile({tile.data(), tile.size()});

    // Should produce 2 plaintexts
    EXPECT_EQ(encoded.size(), 2);
}

TEST_F(TileEncoderTest, RoundTripSmallTile) {
    // Create a small tile
    std::vector<uint8_t> original(1024);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Encode and decode
    auto encoded = encoder_->encode_tile({original.data(), original.size()});
    auto decoded = encoder_->decode_tile(encoded, original.size());

    // Should match
    EXPECT_EQ(decoded.size(), original.size());
    EXPECT_EQ(decoded, original);
}

TEST_F(TileEncoderTest, RoundTripLargeTile) {
    // Create a 30KB tile
    std::vector<uint8_t> original(30720);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (auto& b : original) {
        b = dist(rng_);
    }

    // Encode and decode
    auto encoded = encoder_->encode_tile({original.data(), original.size()});
    auto decoded = encoder_->decode_tile(encoded, original.size());

    // Should match
    EXPECT_EQ(decoded.size(), original.size());
    EXPECT_EQ(decoded, original);
}

TEST_F(TileEncoderTest, BatchEncode) {
    // Create multiple tiles
    const size_t num_tiles = 10;
    const size_t tile_size = 1024;

    std::vector<std::vector<uint8_t>> tiles(num_tiles);
    std::vector<std::span<const uint8_t>> tile_spans;

    for (size_t i = 0; i < num_tiles; ++i) {
        tiles[i].resize(tile_size);
        for (size_t j = 0; j < tile_size; ++j) {
            tiles[i][j] = static_cast<uint8_t>((i + j) & 0xFF);
        }
        tile_spans.emplace_back(tiles[i].data(), tiles[i].size());
    }

    // Batch encode
    auto encoded = encoder_->encode_tiles(tile_spans);

    // Should produce num_tiles plaintexts (each tile fits in 1)
    EXPECT_EQ(encoded.size(), num_tiles);
}

TEST_F(TileEncoderTest, EmptyTile) {
    // Empty tile should produce 0 plaintexts
    std::vector<uint8_t> empty;
    auto encoded = encoder_->encode_tile({empty.data(), empty.size()});
    EXPECT_EQ(encoded.size(), 0);
}
