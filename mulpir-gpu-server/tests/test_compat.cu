#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <tuple>
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute modular inverse: returns x such that a*x = 1 (mod m).
/// Returns 0 if no inverse exists (gcd(a,m) != 1).
inline uint64_t mod_inverse(uint64_t a, uint64_t m) {
    int64_t old_r = static_cast<int64_t>(a);
    int64_t r = static_cast<int64_t>(m);
    int64_t old_s = 1, s = 0;
    while (r != 0) {
        int64_t q = old_r / r;
        std::tie(old_r, r) = std::make_pair(r, old_r - q * r);
        std::tie(old_s, s) = std::make_pair(s, old_s - q * s);
    }
    if (old_r != 1) return 0;
    return static_cast<uint64_t>(
        (old_s % static_cast<int64_t>(m) + static_cast<int64_t>(m)) %
        static_cast<int64_t>(m));
}

/// Encode raw polynomial coefficients into a BFV plaintext, bypassing
/// HEonGPU's batch encoder. This matches fhe.rs's polynomial encoding.
///
/// HEonGPU's HEEncoder is a batch (SIMD) encoder that maps slot values
/// through CRT/NTT. The MulPIR scheme requires polynomial encoding where
/// the coefficient vector IS the polynomial — no permutation or NTT.
inline Plaintext poly_encode(HEContextPtr context,
                             const std::vector<uint64_t>& coeffs) {
    // Initialize via the batch encoder to set the plaintext's internal state
    // (plain_size_, scheme_, storage_type_, etc.)
    HEEncoder encoder(context);
    std::vector<uint64_t> zeros(BFVConfig::POLY_DEGREE, 0);
    Plaintext pt(context);
    encoder.encode(pt, zeros);

    // Overwrite device memory with raw polynomial coefficients
    pt.store_in_device();
    cudaMemcpy(pt.data(), coeffs.data(),
               BFVConfig::POLY_DEGREE * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return pt;
}

/// Decode raw polynomial coefficients from a BFV plaintext, bypassing
/// HEonGPU's batch decoder.
inline std::vector<uint64_t> poly_decode(Plaintext& pt) {
    pt.store_in_device();
    std::vector<uint64_t> coeffs(BFVConfig::POLY_DEGREE);
    cudaMemcpy(coeffs.data(), pt.data(),
               BFVConfig::POLY_DEGREE * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return coeffs;
}

/// Convert raw bytes to polynomial coefficients matching fhe.rs's
/// transcode_from_bytes algorithm.
inline std::vector<uint64_t> bytes_to_poly_coefficients(
    std::span<const uint8_t> bytes) {
    const size_t nbits = BFVConfig::BITS_PER_COEFF;
    const uint64_t mask = (1ULL << nbits) - 1;
    const size_t nelements = BFVConfig::POLY_DEGREE;

    std::vector<uint64_t> out;
    out.reserve(nelements);

    __uint128_t current_value = 0;
    size_t current_value_nbits = 0;
    size_t current_index = 0;

    while (out.size() < nelements) {
        while (current_value_nbits < nbits && current_index < bytes.size()) {
            current_value |=
                static_cast<__uint128_t>(bytes[current_index])
                << current_value_nbits;
            current_value_nbits += 8;
            current_index++;
        }
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

/// Convert polynomial coefficients back to bytes matching fhe.rs's
/// transcode_to_bytes algorithm.
inline std::vector<uint8_t> poly_coefficients_to_bytes(
    std::span<const uint64_t> coeffs,
    size_t output_size) {
    const size_t nbits = BFVConfig::BITS_PER_COEFF;
    const uint64_t mask = (1ULL << nbits) - 1;

    std::vector<uint8_t> out;
    out.reserve(output_size);

    __uint128_t current_value = 0;
    size_t current_value_nbits = 0;
    size_t current_index = 0;

    while (out.size() < output_size) {
        while (current_value_nbits < 8 && current_index < coeffs.size()) {
            const uint64_t coeff = coeffs[current_index] & mask;
            current_value |=
                static_cast<__uint128_t>(coeff) << current_value_nbits;
            current_value_nbits += nbits;
            current_index++;
        }
        if (current_value_nbits >= 8) {
            out.push_back(static_cast<uint8_t>(current_value & 0xFF));
            current_value >>= 8;
            current_value_nbits -= 8;
        } else if (out.size() < output_size) {
            out.push_back(static_cast<uint8_t>(current_value & 0xFF));
            current_value = 0;
            current_value_nbits = 0;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class CompatibilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(TEST_VECTORS_DIR)) {
            GTEST_SKIP() << "Test vectors not found at " << TEST_VECTORS_DIR
                         << ". Run generate_test_vectors first.";
        }

        context_ = create_test_context();
        vectors_ = TestVectors::load(TEST_VECTORS_DIR, context_);

        if (!vectors_.is_valid()) {
            GTEST_SKIP() << "Test vectors failed validation";
        }
    }

    HEContextPtr context_;
    TestVectors vectors_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(CompatibilityTest, TestVectorsLoaded) {
    EXPECT_TRUE(vectors_.is_valid());
    EXPECT_FALSE(vectors_.tiles.empty());
    EXPECT_NE(vectors_.galois_key, nullptr);
    EXPECT_NE(vectors_.relin_key, nullptr);
    EXPECT_NE(vectors_.secret_key, nullptr);
    EXPECT_FALSE(vectors_.query_indices.empty());
    EXPECT_FALSE(vectors_.expected.empty());

    EXPECT_EQ(vectors_.params.poly_degree, BFVConfig::POLY_DEGREE);
    EXPECT_EQ(vectors_.params.plaintext_modulus, BFVConfig::PLAINTEXT_MODULUS);
    EXPECT_EQ(vectors_.params.tile_size, vectors_.tiles[0].size());
    EXPECT_EQ(vectors_.params.num_tiles, vectors_.tiles.size());
    EXPECT_GT(vectors_.params.dim1, 0u);
    EXPECT_GT(vectors_.params.dim2, 0u);
    EXPECT_GT(vectors_.params.expansion_level, 0u);

    std::cout << "Test vectors loaded:" << std::endl;
    std::cout << "  Tiles: " << vectors_.tiles.size() << std::endl;
    std::cout << "  Query indices: " << vectors_.query_indices.size()
              << std::endl;
    std::cout << "  Expected results: " << vectors_.expected.size()
              << std::endl;
    std::cout << "  Dimensions: " << vectors_.params.dim1
              << "x" << vectors_.params.dim2 << std::endl;
    std::cout << "  Expansion level: " << vectors_.params.expansion_level
              << std::endl;
    std::cout << "  Elements per plaintext: "
              << vectors_.params.elements_per_plaintext << std::endl;
}

TEST_F(CompatibilityTest, ModInverseCorrectness) {
    const uint64_t t = BFVConfig::PLAINTEXT_MODULUS;

    EXPECT_EQ(mod_inverse(1, t), 1u);

    const size_t level = vectors_.params.expansion_level;
    const uint64_t power = 1ULL << level;
    const uint64_t inv = mod_inverse(power, t);
    EXPECT_NE(inv, 0u) << "No modular inverse for 2^" << level << " mod " << t;

    const uint64_t product_mod_t =
        static_cast<uint64_t>((__uint128_t(inv) * power) % t);
    EXPECT_EQ(product_mod_t, 1u) << "inv * 2^level should be 1 mod t";

    std::cout << "Modular inverse: 2^" << level << " = " << power
              << ", inv = " << inv << std::endl;
}

TEST_F(CompatibilityTest, EncryptDecryptRoundTrip) {
    // Verify the imported secret key works for encrypt/decrypt via the
    // batch encoder (standard HEonGPU path).
    HEKeyGenerator keygen(context_);
    PublicKey pk(context_);
    keygen.generate_public_key(pk, *vectors_.secret_key);
    cudaDeviceSynchronize();

    HEEncoder encoder(context_);
    HEEncryptor encryptor(context_, pk);
    HEDecryptor decryptor(context_, *vectors_.secret_key);

    std::vector<uint64_t> original(BFVConfig::POLY_DEGREE, 0);
    original[0] = 42;
    original[1] = 100;
    original[BFVConfig::POLY_DEGREE - 1] = 7;

    Plaintext pt(context_);
    encoder.encode(pt, original);

    Ciphertext ct(context_);
    encryptor.encrypt(ct, pt);

    Plaintext pt_dec(context_);
    decryptor.decrypt(pt_dec, ct);

    std::vector<uint64_t> decoded(BFVConfig::POLY_DEGREE);
    encoder.decode(decoded, pt_dec);

    EXPECT_EQ(decoded[0], 42u);
    EXPECT_EQ(decoded[1], 100u);
    EXPECT_EQ(decoded[BFVConfig::POLY_DEGREE - 1], 7u);
    std::cout << "Encrypt/decrypt round-trip: OK" << std::endl;
}

TEST_F(CompatibilityTest, TileEncodeDecodeRoundTrip) {
    // Verify tile polynomial encoding round-trips correctly.
    encoding::TileEncoder tile_encoder(context_);

    const auto& tile = vectors_.tiles[0];
    auto encoded = tile_encoder.encode_tile(
        std::span<const uint8_t>(tile.data(), tile.size()));
    ASSERT_EQ(encoded.size(), 1u)
        << "Expected 1 plaintext for a 20480-byte tile";

    auto decoded = tile_encoder.decode_tile(encoded, tile.size());
    EXPECT_EQ(decoded, tile) << "Tile encode/decode round-trip failed";
    std::cout << "Tile encode/decode round-trip: OK" << std::endl;
}

TEST_F(CompatibilityTest, DiagnosticPolyMultiplyPlain) {
    // Test that polynomial encoding works correctly with HEonGPU's multiply_plain.
    // Encrypt a polynomial with a single nonzero coeff, multiply by a database
    // plaintext, decrypt, and verify.
    //
    // If we encrypt poly [1, 0, 0, ...] and multiply by poly [a0, a1, a2, ...],
    // we should get back [a0, a1, a2, ...] (identity multiplication).

    HEKeyGenerator keygen(context_);
    PublicKey pk(context_);
    keygen.generate_public_key(pk, *vectors_.secret_key);
    cudaDeviceSynchronize();

    HEEncryptor encryptor(context_, pk);
    HEDecryptor decryptor(context_, *vectors_.secret_key);
    HEEncoder encoder(context_);
    HEArithmeticOperator ops(context_, encoder);

    // Create a plaintext polynomial with known coefficients
    std::vector<uint64_t> pt_coeffs(BFVConfig::POLY_DEGREE, 0);
    pt_coeffs[0] = 42;
    pt_coeffs[1] = 100;
    pt_coeffs[2] = 200;
    Plaintext database_pt = poly_encode(context_, pt_coeffs);

    // Create a query polynomial [1, 0, 0, ...]
    std::vector<uint64_t> query_coeffs(BFVConfig::POLY_DEGREE, 0);
    query_coeffs[0] = 1;
    Plaintext query_pt = poly_encode(context_, query_coeffs);

    // Encrypt the query
    Ciphertext query_ct(context_);
    encryptor.encrypt(query_ct, query_pt);

    // Multiply ciphertext by plaintext
    Ciphertext result_ct(context_);
    ops.multiply_plain(query_ct, database_pt, result_ct);

    // Decrypt
    Plaintext result_pt(context_);
    decryptor.decrypt(result_pt, result_ct);

    // Read polynomial coefficients
    auto result_coeffs = poly_decode(result_pt);

    // The result should be [42, 100, 200, 0, 0, ...]
    // (since we multiplied [1, 0, 0, ...] * [42, 100, 200, ...])
    std::cout << "DiagnosticPolyMultiplyPlain result[0..5]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << result_coeffs[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(result_coeffs[0], 42u) << "Coefficient 0 should be 42";
    EXPECT_EQ(result_coeffs[1], 100u) << "Coefficient 1 should be 100";
    EXPECT_EQ(result_coeffs[2], 200u) << "Coefficient 2 should be 200";
    for (size_t i = 3; i < 10; ++i) {
        EXPECT_EQ(result_coeffs[i], 0u) << "Coefficient " << i << " should be 0";
    }
}

TEST_F(CompatibilityTest, DiagnosticExpansion) {
    // Test just the expansion step with a simple input.
    // Encrypt poly [inv, 0, 0, ...] and expand. After expansion, the first
    // expanded ciphertext should decrypt to all-1 (constant polynomial 1),
    // and the rest should be all-0.

    const size_t expansion_level = vectors_.params.expansion_level;
    const uint64_t inv =
        mod_inverse(1ULL << expansion_level, BFVConfig::PLAINTEXT_MODULUS);
    const size_t dim1 = vectors_.params.dim1;
    const size_t dim2 = vectors_.params.dim2;

    HEKeyGenerator keygen(context_);
    PublicKey pk(context_);
    keygen.generate_public_key(pk, *vectors_.secret_key);
    cudaDeviceSynchronize();

    HEEncryptor encryptor(context_, pk);
    HEDecryptor decryptor(context_, *vectors_.secret_key);

    // Create query: coefficient[0] = inv, rest = 0
    // After expansion, this should produce: expanded[0] = Enc(1), rest = Enc(0)
    std::vector<uint64_t> query_coeffs(BFVConfig::POLY_DEGREE, 0);
    query_coeffs[0] = inv;

    Plaintext query_pt = poly_encode(context_, query_coeffs);
    Ciphertext query_ct(context_);
    encryptor.encrypt(query_ct, query_pt);

    // Expand
    pir::QueryExpander expander(context_, *vectors_.galois_key,
                                 PIRDimensions::compute(vectors_.params.num_tiles,
                                                        vectors_.params.tile_size));
    auto expanded = expander.expand(query_ct);

    std::cout << "DiagnosticExpansion: expanded " << expanded.size()
              << " ciphertexts (dim1=" << dim1 << ", dim2=" << dim2 << ")" << std::endl;

    // Decrypt first few and check
    for (size_t i = 0; i < std::min(expanded.size(), size_t{5}); ++i) {
        Plaintext dec_pt(context_);
        decryptor.decrypt(dec_pt, expanded[i]);
        auto coeffs = poly_decode(dec_pt);

        // Count nonzero coefficients
        size_t nonzero = 0;
        for (auto c : coeffs) {
            if (c != 0) nonzero++;
        }

        std::cout << "  expanded[" << i << "]: coeff[0]=" << coeffs[0]
                  << ", nonzero=" << nonzero;
        if (i == 0) {
            std::cout << " (expected: coeff[0]=1, nonzero=1)";
        } else {
            std::cout << " (expected: all zeros)";
        }
        std::cout << std::endl;
    }

    // Verify expanded[0] decrypts to constant 1
    {
        Plaintext dec_pt(context_);
        decryptor.decrypt(dec_pt, expanded[0]);
        auto coeffs = poly_decode(dec_pt);
        EXPECT_EQ(coeffs[0], 1u) << "expanded[0] coefficient 0 should be 1";
        for (size_t i = 1; i < 10; ++i) {
            EXPECT_EQ(coeffs[i], 0u) << "expanded[0] coefficient " << i << " should be 0";
        }
    }
}

TEST_F(CompatibilityTest, SharedSecretKeyPIR) {
    // Full end-to-end PIR test using the imported fhe.rs secret key.
    //
    // Steps:
    //   1. Load the database tiles into DatabaseManager
    //   2. Generate a public key from the imported secret key
    //   3. Create the PIR engine with imported Galois/Relin keys
    //   4. For each query index:
    //      a. Create the query matching fhe.rs encoding convention
    //      b. Process through the PIR pipeline
    //      c. Decrypt the result
    //      d. Decode and extract the target tile
    //      e. Compare against expected bytes

    const auto& params = vectors_.params;
    const size_t tile_size = params.tile_size;
    const size_t elements_per_plaintext = params.elements_per_plaintext;
    const size_t dim1 = params.dim1;
    const size_t dim2 = params.dim2;
    const size_t expansion_level = params.expansion_level;

    std::cout << "\n=== SharedSecretKeyPIR e2e Test ===" << std::endl;
    std::cout << "  dim1=" << dim1 << " dim2=" << dim2
              << " expansion_level=" << expansion_level << std::endl;
    std::cout << "  tile_size=" << tile_size
              << " elements_per_plaintext=" << elements_per_plaintext
              << std::endl;

    // 1. Load database (uses polynomial encoding via TileEncoder)
    std::cout << "Loading database (" << vectors_.tiles.size()
              << " tiles)..." << std::endl;
    ServerConfig server_config;
    server_config.tile_size_bytes = tile_size;
    database::DatabaseManager db(context_, server_config);
    db.load_database(vectors_.tiles);
    ASSERT_TRUE(db.is_ready()) << "Database failed to load";

    const auto& db_dims = db.dimensions();
    EXPECT_EQ(db_dims.dim1, dim1);
    EXPECT_EQ(db_dims.dim2, dim2);
    EXPECT_EQ(db_dims.elements_per_plaintext, elements_per_plaintext);

    // 2. Generate public key from imported secret key
    std::cout << "Generating public key..." << std::endl;
    HEKeyGenerator keygen(context_);
    PublicKey pk(context_);
    keygen.generate_public_key(pk, *vectors_.secret_key);
    cudaDeviceSynchronize();

    // 3. Create PIR engine
    std::cout << "Creating PIR engine..." << std::endl;
    pir::PIREngine engine(
        context_,
        *vectors_.galois_key,
        *vectors_.relin_key,
        db);
    ASSERT_TRUE(engine.is_ready()) << "PIR engine not ready";

    // Selection value: modular inverse of 2^expansion_level
    const uint64_t inv =
        mod_inverse(1ULL << expansion_level, BFVConfig::PLAINTEXT_MODULUS);
    ASSERT_NE(inv, 0u) << "Modular inverse does not exist";

    HEEncryptor encryptor(context_, pk);
    HEDecryptor decryptor(context_, *vectors_.secret_key);

    size_t num_passed = 0;
    size_t num_tested = 0;

    // 4. For each query index, run the full PIR pipeline
    for (size_t tile_index : vectors_.query_indices) {
        auto expected_it = vectors_.expected.find(tile_index);
        if (expected_it == vectors_.expected.end()) {
            std::cout << "  Skipping tile " << tile_index
                      << " (no expected result)" << std::endl;
            continue;
        }
        const auto& expected = expected_it->second;
        ++num_tested;

        std::cout << "  Testing tile_index=" << tile_index << "..."
                  << std::flush;

        // 4a. Construct the query using polynomial encoding
        const size_t query_index = tile_index / elements_per_plaintext;
        const size_t row = query_index / dim2;
        const size_t col = query_index % dim2;

        std::vector<uint64_t> query_coeffs(BFVConfig::POLY_DEGREE, 0);
        query_coeffs[row] = inv;
        query_coeffs[dim1 + col] = inv;

        // Polynomial-encode and encrypt the query
        Plaintext query_pt = poly_encode(context_, query_coeffs);
        Ciphertext query_ct(context_);
        encryptor.encrypt(query_ct, query_pt);

        // 4b. Process through PIR pipeline
        PIRQuery pir_query;
        pir_query.encrypted_query = std::move(query_ct);
        pir_query.metadata.query_id = tile_index;

        auto response = engine.process_query(pir_query);

        const auto& stats = engine.last_query_stats();
        std::cout << " PIR " << stats.total_ms << "ms";

        // 4c. Decrypt the result
        Plaintext result_pt(context_);
        decryptor.decrypt(result_pt, response.encrypted_result);

        // 4d. Read raw polynomial coefficients and convert to bytes
        auto result_coeffs = poly_decode(result_pt);
        auto row_bytes = poly_coefficients_to_bytes(
            result_coeffs,
            elements_per_plaintext * tile_size);

        // 4e. Extract the specific tile at offset_in_row
        const size_t offset = expected.offset_in_row;
        const size_t start = offset * tile_size;
        const size_t end = start + tile_size;

        ASSERT_LE(end, row_bytes.size())
            << "Row bytes too short at offset " << offset;

        std::vector<uint8_t> actual_tile(
            row_bytes.begin() + static_cast<ptrdiff_t>(start),
            row_bytes.begin() + static_cast<ptrdiff_t>(end));

        // Compare against expected tile bytes
        if (actual_tile == expected.tile_bytes) {
            std::cout << " PASS" << std::endl;
            ++num_passed;
        } else {
            std::cout << " FAIL" << std::endl;

            for (size_t b = 0; b < tile_size; ++b) {
                if (actual_tile[b] != expected.tile_bytes[b]) {
                    std::cout << "    First diff at byte " << b
                              << ": got 0x" << std::hex
                              << static_cast<int>(actual_tile[b])
                              << " expected 0x"
                              << static_cast<int>(expected.tile_bytes[b])
                              << std::dec << std::endl;
                    break;
                }
            }

            EXPECT_EQ(actual_tile, expected.tile_bytes)
                << "Tile mismatch at index " << tile_index
                << " (offset_in_row=" << offset << ")";
        }
    }

    std::cout << "\nResults: " << num_passed << "/" << num_tested
              << " tiles matched" << std::endl;

    EXPECT_GT(num_tested, 0u) << "No query indices were tested";
    EXPECT_EQ(num_passed, num_tested) << "Some tiles did not match";
}
