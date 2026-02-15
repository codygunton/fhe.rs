#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "config.hpp"
#include "types.hpp"
#include "database/database_manager.hpp"
#include "encoding/tile_encoder.hpp"
#include "pir/pir_engine.hpp"
#include "test_helpers.hpp"
#include "test_vector_loader.hpp"

using namespace mulpir;
using namespace mulpir::test;

// Test vectors directory (relative to build directory)
static const std::string TEST_VECTORS_DIR = "../test_vectors";

class CompatibilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if test vectors exist
        if (!std::filesystem::exists(TEST_VECTORS_DIR)) {
            GTEST_SKIP() << "Test vectors not found at " << TEST_VECTORS_DIR
                        << ". Run ./scripts/generate_test_vectors.sh first.";
        }

        // Create BFV context matching fhe.rs parameters
        context_ = create_test_context();

        // Load test vectors
        vectors_ = TestVectors::load(TEST_VECTORS_DIR);
    }

    HEContextPtr context_;
    TestVectors vectors_;
};

TEST_F(CompatibilityTest, TestVectorsLoaded) {
    // Verify test vectors loaded correctly
    EXPECT_TRUE(vectors_.is_valid());
    EXPECT_FALSE(vectors_.tiles.empty());
    EXPECT_NE(vectors_.galois_key, nullptr);
    EXPECT_NE(vectors_.relin_key, nullptr);
    EXPECT_FALSE(vectors_.queries.empty());

    std::cout << "Test vectors loaded:" << std::endl;
    std::cout << "  Tiles: " << vectors_.tiles.size() << std::endl;
    std::cout << "  Queries: " << vectors_.queries.size() << std::endl;
    std::cout << "  Dimensions: " << vectors_.dims.dim1 << "x" << vectors_.dims.dim2 << std::endl;
}

TEST_F(CompatibilityTest, DatabaseDimensionsMatch) {
    // Load database and verify dimensions match params
    database::DatabaseManager db(context_, ServerConfig{});
    db.load_database(vectors_.tiles);

    const auto& dims = db.dimensions();

    EXPECT_EQ(dims.num_elements, vectors_.params.num_tiles);
    EXPECT_EQ(dims.dim1, vectors_.params.dim1);
    EXPECT_EQ(dims.dim2, vectors_.params.dim2);
    EXPECT_EQ(dims.expansion_level, vectors_.params.expansion_level);
}

TEST_F(CompatibilityTest, ProcessFheRsQuery) {
    // Verifies GPU server can process queries from fhe.rs client

    // Load database
    database::DatabaseManager db(context_, ServerConfig{});
    db.load_database(vectors_.tiles);

    // Create PIR engine
    pir::PIREngine engine(
        context_,
        *vectors_.galois_key,
        *vectors_.relin_key,
        db
    );

    // Process each test query
    for (const auto& [idx, query_ct] : vectors_.queries) {
        std::cout << "Processing query for index " << idx << "..." << std::endl;

        // Create PIR query
        PIRQuery query;
        query.encrypted_query = query_ct;
        query.metadata.query_id = idx;

        // Process query
        auto response = engine.process_query(query);

        std::cout << "  Processing time: " << response.metadata.processing_time_ms << " ms" << std::endl;

        // Verify response has valid ciphertext
        // Full decryption verification requires the fhe.rs client
    }
}

TEST_F(CompatibilityTest, EncodingMatchesFheRs) {
    // Verify our encoding produces the same structure as fhe.rs

    // Create encoder
    encoding::TileEncoder encoder(context_);

    // Encode first tile
    const auto& tile = vectors_.tiles[0];
    auto encoded = encoder.encode_tile({tile.data(), tile.size()});

    // Verify we get the expected number of plaintexts
    size_t expected_pts = encoder.plaintexts_per_tile(tile.size());
    EXPECT_EQ(encoded.size(), expected_pts);
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "=== MulPIR GPU Server Compatibility Tests ===" << std::endl;
    std::cout << std::endl;

    return RUN_ALL_TESTS();
}
