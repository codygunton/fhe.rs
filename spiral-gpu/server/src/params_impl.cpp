// SpiralParams::to_json() and select_params() implementation.

#include "params.hpp"

#include <cassert>
#include <nlohmann/json.hpp>

std::string SpiralParams::to_json() const {
    nlohmann::json j;
    j["n"]            = N;
    j["nu_1"]         = nu_1;
    j["nu_2"]         = nu_2;
    j["p"]            = P;
    j["q2_bits"]      = Q2_BITS;
    j["t_gsw"]        = T_GSW;
    j["t_conv"]       = T_CONV;
    j["t_exp_left"]   = T_EXP_LEFT;
    j["t_exp_right"]  = T_EXP_RIGHT;
    j["instances"]    = instances;
    j["db_item_size"] = db_item_size;
    return j.dump();
}

// select_params() — mirrors spiral-cpu/server select_params_json() exactly.
//
// With p=256 (8 bits/coeff) and poly_len=2048: 2048 bytes per chunk.
// chunks_needed = ceil(tile_size / 2048)
// instances     = ceil(chunks_needed / n^2)   (n^2 = 4)
//
// nu_1 = 9 (left dimension = 512 rows, matches Blyss v1 production shape).
// nu_2 chosen so total capacity 2^(9+nu_2) >= num_tiles.
SpiralParams select_params(size_t num_tiles, size_t tile_size) {
    constexpr size_t BYTES_PER_CHUNK = SPIRAL_POLY_LEN;  // 2048 bytes (poly_len * 8 bits / 8)
    size_t chunks_needed = (tile_size + BYTES_PER_CHUNK - 1) / BYTES_PER_CHUNK;
    uint32_t instances = static_cast<uint32_t>((chunks_needed + 3) / 4);  // ceil / n^2

    constexpr uint32_t nu_1 = 9;
    uint32_t nu_2;
    if (num_tiles <= (1u << (nu_1 + 2))) {
        nu_2 = 2;  // 2^11 = 2048 items
    } else if (num_tiles <= (1u << (nu_1 + 4))) {
        nu_2 = 4;  // 2^13 = 8192 items
    } else {
        nu_2 = 6;  // 2^15 = 32768 items
    }

    return SpiralParams(nu_1, nu_2, instances, static_cast<uint32_t>(tile_size));
}
