#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "config.hpp"
#include "types.hpp"
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
                        << ". Run generate_test_vectors first.";
        }

        // Create BFV context matching fhe.rs parameters
        context_ = create_test_context();

        // Load test vectors (needs context for SecretKey construction)
        vectors_ = TestVectors::load(TEST_VECTORS_DIR, context_);
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
    EXPECT_NE(vectors_.secret_key, nullptr);
    EXPECT_FALSE(vectors_.query_indices.empty());

    std::cout << "Test vectors loaded:" << std::endl;
    std::cout << "  Tiles: " << vectors_.tiles.size() << std::endl;
    std::cout << "  Query indices: " << vectors_.query_indices.size() << std::endl;
    std::cout << "  Expected results: " << vectors_.expected.size() << std::endl;
    std::cout << "  Dimensions: " << vectors_.params.dim1
              << "x" << vectors_.params.dim2 << std::endl;
    std::cout << "  Expansion level: " << vectors_.params.expansion_level << std::endl;
}

// Placeholder: full e2e tests will be added in Task 4.

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "=== MulPIR GPU Server Compatibility Tests ===" << std::endl;
    std::cout << std::endl;

    return RUN_ALL_TESTS();
}
