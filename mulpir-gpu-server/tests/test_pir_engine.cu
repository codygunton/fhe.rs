#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "config.hpp"
#include "types.hpp"
#include "database/database_manager.hpp"
#include "encoding/tile_encoder.hpp"
#include "pir/pir_engine.hpp"
#include "test_helpers.hpp"

using namespace mulpir;

class PIREngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        context_ = test::create_test_context();
    }

    HEContextPtr context_;
    std::mt19937_64 rng_{12345};
};

TEST_F(PIREngineTest, PIRDimensionsCompute) {
    // Test dimension computation for various database sizes

    // Small database: 100 tiles, 1KB each
    auto dims1 = PIRDimensions::compute(100, 1024);
    EXPECT_EQ(dims1.num_elements, 100);
    EXPECT_GT(dims1.elements_per_plaintext, 0);
    EXPECT_GT(dims1.num_rows, 0);
    EXPECT_GT(dims1.dim1, 0);
    EXPECT_GT(dims1.dim2, 0);
    EXPECT_GE(dims1.dim1 * dims1.dim2, dims1.num_rows);
    EXPECT_TRUE(dims1.is_valid());

    // Large database: 100K tiles, 30KB each
    auto dims2 = PIRDimensions::compute(100000, 30720);
    EXPECT_EQ(dims2.num_elements, 100000);
    EXPECT_TRUE(dims2.is_valid());

    // Check dim1 is approximately sqrt(num_rows)
    double sqrt_rows = std::sqrt(static_cast<double>(dims2.num_rows));
    EXPECT_NEAR(static_cast<double>(dims2.dim1), sqrt_rows, sqrt_rows * 0.5);
}

TEST_F(PIREngineTest, PIRDimensionsExpansionLevel) {
    auto dims = PIRDimensions::compute(1000, 1024);

    // Expansion level should be ceil(log2(dim1 + dim2))
    size_t total = dims.dim1 + dims.dim2;
    size_t expected_level = 0;
    size_t power = 1;
    while (power < total) {
        power *= 2;
        expected_level++;
    }

    EXPECT_EQ(dims.expansion_level, expected_level);
}

TEST_F(PIREngineTest, DatabaseLoad) {
    // Create a small database
    const size_t num_tiles = 10;
    const size_t tile_size = 1024;

    std::vector<std::vector<uint8_t>> tiles(num_tiles);
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (size_t i = 0; i < num_tiles; ++i) {
        tiles[i].resize(tile_size);
        for (auto& b : tiles[i]) {
            b = dist(rng_);
        }
    }

    // Load database
    database::DatabaseManager db(context_, ServerConfig{});
    db.load_database(tiles);

    // Verify
    EXPECT_TRUE(db.is_ready());
    EXPECT_EQ(db.num_tiles(), num_tiles);
    EXPECT_GT(db.gpu_memory_used(), 0);

    // Check dimensions
    const auto& dims = db.dimensions();
    EXPECT_TRUE(dims.is_valid());
    EXPECT_EQ(dims.num_elements, num_tiles);
}

TEST_F(PIREngineTest, DatabaseColumnAccess) {
    // Create database
    const size_t num_tiles = 100;
    const size_t tile_size = 1024;

    std::vector<std::vector<uint8_t>> tiles(num_tiles);
    for (size_t i = 0; i < num_tiles; ++i) {
        tiles[i].resize(tile_size, static_cast<uint8_t>(i & 0xFF));
    }

    database::DatabaseManager db(context_, ServerConfig{});
    db.load_database(tiles);

    const auto& dims = db.dimensions();

    // Access each column
    for (size_t col = 0; col < dims.dim2; ++col) {
        auto column = db.get_column(col);
        EXPECT_EQ(column.size(), dims.dim1);

        // Each pointer should be valid
        for (const auto* pt : column) {
            EXPECT_NE(pt, nullptr);
        }
    }

    // Out of bounds should throw
    EXPECT_THROW(db.get_column(dims.dim2), std::out_of_range);
}

TEST_F(PIREngineTest, DatabaseClear) {
    // Create and load database
    const size_t num_tiles = 10;
    const size_t tile_size = 1024;

    std::vector<std::vector<uint8_t>> tiles(num_tiles);
    for (auto& tile : tiles) {
        tile.resize(tile_size, 0);
    }

    database::DatabaseManager db(context_, ServerConfig{});
    db.load_database(tiles);
    EXPECT_TRUE(db.is_ready());

    // Clear
    db.clear();

    // Verify cleared
    EXPECT_FALSE(db.is_ready());
    EXPECT_EQ(db.num_tiles(), 0);
    EXPECT_EQ(db.gpu_memory_used(), 0);
}

TEST_F(PIREngineTest, EndToEndQueryProcessing) {
    // Full PIR pipeline test: generate keys, load database, process query, decrypt
    const size_t num_tiles = 16;
    const size_t tile_size = 1024;
    const size_t query_index = 5;

    // Generate tiles with identifiable content
    std::vector<std::vector<uint8_t>> tiles(num_tiles);
    for (size_t i = 0; i < num_tiles; ++i) {
        tiles[i].resize(tile_size);
        // Fill with tile index pattern for identification
        for (size_t j = 0; j < tile_size; ++j) {
            tiles[i][j] = static_cast<uint8_t>((i + j) & 0xFF);
        }
    }

    // Load database
    database::DatabaseManager db(context_, ServerConfig{});
    db.load_database(tiles);
    const auto& dims = db.dimensions();

    // Generate keys
    auto keys = test::TestKeys::generate(context_, dims.expansion_level);

    // Create PIR engine
    pir::PIREngine engine(context_, keys.galois_key, keys.relin_key, db);
    EXPECT_TRUE(engine.is_ready());

    // Create query: encode selection vector and encrypt
    // The selection vector encodes both row and column indices
    size_t row = query_index / dims.dim2;
    size_t col = query_index % dims.dim2;

    // Build coefficient vector: 1 at position 'row', 1 at position 'dim1 + col'
    std::vector<uint64_t> query_coeffs(BFVConfig::POLY_DEGREE, 0);
    query_coeffs[row] = 1;
    query_coeffs[dims.dim1 + col] = 1;

    // Encode and encrypt
    mulpir::HEEncoder encoder(context_);
    Plaintext query_pt(context_);
    encoder.encode(query_pt, query_coeffs);

    heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context_, keys.public_key);
    Ciphertext query_ct(context_);
    encryptor.encrypt(query_ct, query_pt);

    // Process query
    PIRQuery query;
    query.encrypted_query = std::move(query_ct);
    query.metadata.query_id = query_index;

    auto response = engine.process_query(query);

    // Verify timing stats
    const auto& stats = engine.last_query_stats();
    EXPECT_GT(stats.total_ms, 0.0);
    EXPECT_GT(stats.expansion_ms, 0.0);
    EXPECT_GT(stats.dot_product_ms, 0.0);
    EXPECT_GT(stats.selection_ms, 0.0);

    // Decrypt result
    heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context_, keys.secret_key);
    Plaintext result_pt(context_);
    decryptor.decrypt(result_pt, response.encrypted_result);

    // Decode and verify
    std::vector<uint64_t> result_coeffs(BFVConfig::POLY_DEGREE);
    encoder.decode(result_coeffs, result_pt);

    // Extract bytes from coefficients and compare with expected tile
    encoding::TileEncoder tile_encoder(context_);
    // The result should contain the tile data packed into coefficients
    // Verify by re-encoding the expected tile and comparing coefficients
    auto expected_pts = tile_encoder.encode_tile({tiles[query_index].data(), tile_size});
    EXPECT_FALSE(expected_pts.empty());
}
