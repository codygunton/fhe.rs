#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "types.hpp"
#include "test_helpers.hpp"

namespace mulpir::test {

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external dependency)
// ---------------------------------------------------------------------------

namespace json_detail {

/// Extract the numeric value for a given key from a flat JSON string.
/// Works for integer values like "dim1": 10
inline uint64_t parse_uint64(const std::string& json, const std::string& key) {
    const std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) {
        throw std::runtime_error("JSON key not found: " + key);
    }
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed JSON for key: " + key);
    }
    ++pos;
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) {
        ++pos;
    }
    return std::stoull(json.substr(pos));
}

/// Extract an array of uint64_t values for a given key.
/// Works for arrays like "query_indices": [0, 1, 10, 42, 99]
inline std::vector<uint64_t> parse_uint64_array(const std::string& json,
                                                 const std::string& key) {
    const std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) {
        throw std::runtime_error("JSON key not found: " + key);
    }
    auto bracket_start = json.find('[', pos);
    auto bracket_end = json.find(']', bracket_start);
    if (bracket_start == std::string::npos || bracket_end == std::string::npos) {
        throw std::runtime_error("Malformed JSON array for key: " + key);
    }

    std::vector<uint64_t> result;
    std::string inner = json.substr(bracket_start + 1,
                                    bracket_end - bracket_start - 1);
    std::istringstream ss(inner);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        result.push_back(std::stoull(token.substr(start)));
    }
    return result;
}

}  // namespace json_detail

// ---------------------------------------------------------------------------
// TestParams — parameters loaded from params.json
// ---------------------------------------------------------------------------

struct TestParams {
    size_t poly_degree = 0;
    uint64_t plaintext_modulus = 0;
    std::vector<uint64_t> q_primes;
    size_t num_tiles = 0;
    size_t tile_size = 0;
    size_t elements_per_plaintext = 0;
    size_t num_rows = 0;
    size_t dim1 = 0;
    size_t dim2 = 0;
    size_t expansion_level = 0;
};

// ---------------------------------------------------------------------------
// TestVectors — everything needed for e2e compatibility tests
// ---------------------------------------------------------------------------

struct TestVectors {
    /// PIR and BFV parameters from params.json.
    TestParams params;

    /// Raw tile bytes loaded from tiles.bin.
    std::vector<std::vector<uint8_t>> tiles;

    /// Secret key reconstructed from raw int coefficients.
    std::unique_ptr<SecretKey> secret_key;

    /// Galois key generated from the imported secret key.
    std::unique_ptr<GaloisKey> galois_key;

    /// Relinearization key generated from the imported secret key.
    std::unique_ptr<RelinKey> relin_key;

    /// Expected result for a single query index.
    struct ExpectedResult {
        size_t offset_in_row;
        std::vector<uint8_t> tile_bytes;
    };

    /// Expected results keyed by query index.
    std::map<size_t, ExpectedResult> expected;

    /// Query indices to test (from params.json).
    std::vector<size_t> query_indices;

    /// Check that the minimum required data was loaded.
    bool is_valid() const {
        return !tiles.empty() && secret_key && galois_key && relin_key &&
               params.dim1 > 0 && params.dim2 > 0;
    }

    /// Load test vectors from a directory.
    /// The caller must supply a fully-initialized HEContext because it is
    /// needed to construct the SecretKey and derive Galois/Relin keys.
    /// Returns an empty (invalid) TestVectors if the directory does not exist.
    static TestVectors load(const std::string& dir, HEContextPtr context) {
        TestVectors tv;
        namespace fs = std::filesystem;

        if (!fs::exists(dir)) {
            return tv;
        }

        // ---------------------------------------------------------------
        // 1. params.json
        // ---------------------------------------------------------------
        {
            auto path = fs::path(dir) / "params.json";
            if (!fs::exists(path)) {
                throw std::runtime_error("params.json not found in " + dir);
            }
            std::ifstream f(path);
            if (!f) {
                throw std::runtime_error("Failed to open " + path.string());
            }
            std::string json((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

            tv.params.poly_degree =
                json_detail::parse_uint64(json, "poly_degree");
            tv.params.plaintext_modulus =
                json_detail::parse_uint64(json, "plaintext_modulus");
            tv.params.num_tiles =
                json_detail::parse_uint64(json, "num_tiles");
            tv.params.tile_size =
                json_detail::parse_uint64(json, "tile_size");
            tv.params.elements_per_plaintext =
                json_detail::parse_uint64(json, "elements_per_plaintext");
            tv.params.num_rows =
                json_detail::parse_uint64(json, "num_rows");
            tv.params.dim1 =
                json_detail::parse_uint64(json, "dim1");
            tv.params.dim2 =
                json_detail::parse_uint64(json, "dim2");
            tv.params.expansion_level =
                json_detail::parse_uint64(json, "expansion_level");

            tv.params.q_primes =
                json_detail::parse_uint64_array(json, "q_primes");

            auto qi_raw =
                json_detail::parse_uint64_array(json, "query_indices");
            tv.query_indices.reserve(qi_raw.size());
            for (auto v : qi_raw) {
                tv.query_indices.push_back(static_cast<size_t>(v));
            }
        }

        // ---------------------------------------------------------------
        // 2. tiles.bin
        // ---------------------------------------------------------------
        {
            auto path = fs::path(dir) / "tiles.bin";
            if (!fs::exists(path)) {
                throw std::runtime_error("tiles.bin not found in " + dir);
            }
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                throw std::runtime_error("Failed to open " + path.string());
            }

            uint64_t num_tiles = 0;
            uint64_t tile_size = 0;
            f.read(reinterpret_cast<char*>(&num_tiles), 8);
            f.read(reinterpret_cast<char*>(&tile_size), 8);

            if (num_tiles != tv.params.num_tiles) {
                throw std::runtime_error(
                    "tiles.bin num_tiles (" + std::to_string(num_tiles) +
                    ") != params.json num_tiles (" +
                    std::to_string(tv.params.num_tiles) + ")");
            }
            if (tile_size != tv.params.tile_size) {
                throw std::runtime_error(
                    "tiles.bin tile_size (" + std::to_string(tile_size) +
                    ") != params.json tile_size (" +
                    std::to_string(tv.params.tile_size) + ")");
            }

            tv.tiles.resize(num_tiles);
            for (size_t i = 0; i < num_tiles; ++i) {
                tv.tiles[i].resize(tile_size);
                f.read(reinterpret_cast<char*>(tv.tiles[i].data()),
                       static_cast<std::streamsize>(tile_size));
            }
        }

        // ---------------------------------------------------------------
        // 3. secret_key.bin  (raw i64 coefficients -> SecretKey via RNS+NTT)
        // ---------------------------------------------------------------
        {
            auto path = fs::path(dir) / "secret_key.bin";
            if (!fs::exists(path)) {
                throw std::runtime_error("secret_key.bin not found in " + dir);
            }
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                throw std::runtime_error("Failed to open " + path.string());
            }

            uint64_t num_coeffs = 0;
            f.read(reinterpret_cast<char*>(&num_coeffs), 8);

            if (num_coeffs != tv.params.poly_degree) {
                throw std::runtime_error(
                    "secret_key.bin num_coeffs (" +
                    std::to_string(num_coeffs) +
                    ") != poly_degree (" +
                    std::to_string(tv.params.poly_degree) + ")");
            }

            // Read i64 coefficients and cast to int (safe: CBD values ~ -9..+9)
            std::vector<int> sk_coeffs(num_coeffs);
            for (size_t i = 0; i < num_coeffs; ++i) {
                int64_t val = 0;
                f.read(reinterpret_cast<char*>(&val), 8);
                sk_coeffs[i] = static_cast<int>(val);
            }

            // Construct SecretKey — handles RNS decomposition + forward NTT
            tv.secret_key = std::make_unique<SecretKey>(
                sk_coeffs, context, cudaStream_t{0});
            cudaDeviceSynchronize();
        }

        // ---------------------------------------------------------------
        // 4. Generate Galois and Relin keys from the imported secret key
        // ---------------------------------------------------------------
        {
            HEKeyGenerator keygen(context);

            auto galois_elts =
                expansion_galois_elements(tv.params.expansion_level);

            tv.relin_key = std::make_unique<RelinKey>(context);
            keygen.generate_relin_key(*tv.relin_key, *tv.secret_key);

            tv.galois_key =
                std::make_unique<GaloisKey>(context, galois_elts);
            keygen.generate_galois_key(*tv.galois_key, *tv.secret_key);

            cudaDeviceSynchronize();
        }

        // ---------------------------------------------------------------
        // 5. expected_{idx}.bin files
        // ---------------------------------------------------------------
        for (size_t idx : tv.query_indices) {
            auto path =
                fs::path(dir) / ("expected_" + std::to_string(idx) + ".bin");
            if (!fs::exists(path)) {
                // Not fatal — the test can skip indices without expectations
                continue;
            }
            std::ifstream f(path, std::ios::binary);
            if (!f) continue;

            ExpectedResult er;
            uint64_t offset = 0;
            f.read(reinterpret_cast<char*>(&offset), 8);
            er.offset_in_row = static_cast<size_t>(offset);

            // Remaining bytes are the raw tile data
            er.tile_bytes.assign(std::istreambuf_iterator<char>(f),
                                 std::istreambuf_iterator<char>());

            tv.expected[idx] = std::move(er);
        }

        return tv;
    }
};

}  // namespace mulpir::test
