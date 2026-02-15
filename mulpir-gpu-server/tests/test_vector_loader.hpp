#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace mulpir::test {

/// Parameters loaded from test vectors.
struct TestParams {
    size_t num_tiles = 0;
    size_t tile_size = 0;
    size_t dim1 = 0;
    size_t dim2 = 0;
    size_t expansion_level = 0;
};

/// Test vectors loaded from disk.
struct TestVectors {
    TestParams params;
    PIRDimensions dims;
    std::vector<std::vector<uint8_t>> tiles;
    std::unique_ptr<GaloisKey> galois_key;
    std::unique_ptr<RelinKey> relin_key;
    std::map<size_t, Ciphertext> queries;
    std::map<size_t, std::vector<uint8_t>> expected;

    bool is_valid() const {
        return !tiles.empty() && galois_key && relin_key;
    }

    std::vector<size_t> query_indices() const {
        std::vector<size_t> indices;
        for (const auto& [idx, _] : queries) {
            indices.push_back(idx);
        }
        return indices;
    }

    /// Load test vectors from directory.
    /// Returns empty TestVectors if directory doesn't exist.
    static TestVectors load(const std::string& dir) {
        TestVectors tv;
        namespace fs = std::filesystem;

        if (!fs::exists(dir)) {
            return tv;
        }

        // Load tiles
        auto tiles_path = fs::path(dir) / "tiles.bin";
        if (fs::exists(tiles_path)) {
            std::ifstream f(tiles_path, std::ios::binary);
            if (f) {
                // Read header: num_tiles (8 bytes) + tile_size (8 bytes)
                uint64_t num_tiles = 0, tile_size = 0;
                f.read(reinterpret_cast<char*>(&num_tiles), 8);
                f.read(reinterpret_cast<char*>(&tile_size), 8);

                tv.params.num_tiles = num_tiles;
                tv.params.tile_size = tile_size;

                tv.tiles.resize(num_tiles);
                for (size_t i = 0; i < num_tiles; ++i) {
                    tv.tiles[i].resize(tile_size);
                    f.read(reinterpret_cast<char*>(tv.tiles[i].data()), tile_size);
                }
            }
        }

        // Load params
        auto params_path = fs::path(dir) / "params.bin";
        if (fs::exists(params_path)) {
            std::ifstream f(params_path, std::ios::binary);
            if (f) {
                f.read(reinterpret_cast<char*>(&tv.params.dim1), 8);
                f.read(reinterpret_cast<char*>(&tv.params.dim2), 8);
                f.read(reinterpret_cast<char*>(&tv.params.expansion_level), 8);
            }
        }

        tv.dims.dim1 = tv.params.dim1;
        tv.dims.dim2 = tv.params.dim2;
        tv.dims.expansion_level = tv.params.expansion_level;
        tv.dims.num_elements = tv.params.num_tiles;

        return tv;
    }
};

}  // namespace mulpir::test
