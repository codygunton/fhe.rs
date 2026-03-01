// Database loading and GPU upload for Spiral PIR.
//
// Mirrors spiral-rs server.rs load_db_from_seek() and load_item_from_seek().
// For each (instance, trial, item): reads 20-bit coefficients from the flat
// tile file, recenters mod pt_modulus, NTTs, CRT-packs, reorders by calc_index,
// then copies the full preprocessed database to GPU VRAM.

#include "types.hpp"
#include "params.hpp"
#include "host_ntt.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

// ── Constants ─────────────────────────────────────────────────────────────────

// Plaintext bits per coefficient = ceil(log2(P)) = ceil(log2(256)) = 8.
// BUT the DB encoding uses logp = ceil(log2(pt_modulus)) where pt_modulus = 256.
// Per spiral-rs: logp = ceil(log2(256)) = 8.
static constexpr uint32_t LOGP = 8;  // bits per DB coefficient

// Large combined modulus for recenter_mod:
//   spiral-rs params.modulus = q0 * q1 = MODULUS_0 * MODULUS_1
static constexpr uint64_t LARGE_MOD = SpiralParams::MODULUS_0 * SpiralParams::MODULUS_1;

// ── Helper: read_arbitrary_bits ───────────────────────────────────────────────
// Direct port of spiral-rs util.rs read_arbitrary_bits().
static uint64_t read_arbitrary_bits(const uint8_t* data, size_t bit_offs, uint32_t num_bits) {
    const size_t word_off = bit_offs / 64;
    const size_t bit_off_within_word = bit_offs % 64;
    if ((bit_off_within_word + num_bits) <= 64) {
        const size_t idx = word_off * 8;
        uint64_t val;
        memcpy(&val, data + idx, 8);
        return (val >> bit_off_within_word) & ((1ULL << num_bits) - 1ULL);
    } else {
        const size_t idx = word_off * 8;
        unsigned __int128 val;
        memcpy(&val, data + idx, 16);
        return static_cast<uint64_t>(
            (val >> bit_off_within_word) & (((unsigned __int128)1 << num_bits) - 1));
    }
}

// ── Helper: recenter_mod ──────────────────────────────────────────────────────
// Mirrors spiral-rs arith.rs recenter_mod():
//   Signed-centers val in Z_{pt_mod}, then lifts to Z_{mod}.
//   If val > pt_mod/2: treat as -(pt_mod - val), return mod - (pt_mod - val)
//   else:              return val
static uint64_t recenter_mod(uint64_t val, uint64_t pt_mod, uint64_t mod) {
    if (val > pt_mod / 2) {
        return mod - (pt_mod - val);
    }
    return val;
}

// ── Helper: Barrett reduce for host ──────────────────────────────────────────
static uint64_t host_barrett(uint64_t val, uint64_t mod, uint64_t cr1) {
    uint64_t q = static_cast<uint64_t>(
        (static_cast<unsigned __int128>(val) * cr1) >> 64);
    uint64_t r = val - q * mod;
    if (r >= mod) r -= mod;
    return r;
}

// ── load_db_to_gpu ────────────────────────────────────────────────────────────

DeviceDB load_db_to_gpu(const uint8_t* raw_tiles, size_t file_size,
                        const SpiralParams& p) {
    const uint32_t N         = SpiralParams::POLY_LEN;
    const uint32_t instances = p.instances;
    const uint32_t trials    = SpiralParams::N * SpiralParams::N;  // n*n = 4
    const uint32_t dim0      = p.dim0();        // 512
    const uint32_t num_per   = p.num_per();     // 16
    const uint32_t num_items = p.num_items();   // 8192

    // Per spiral-rs load_item_from_seek():
    //   chunks = instances * trials
    //   bytes_per_chunk = ceil(db_item_size / chunks)
    //   logp = ceil(log2(pt_modulus)) = 8
    //   modp_words_per_chunk = ceil(bytes_per_chunk*8 / logp) ≤ N
    const uint32_t chunks = instances * trials;
    const uint32_t bytes_per_chunk =
        static_cast<uint32_t>(std::ceil(static_cast<double>(p.db_item_size) / chunks));
    const uint32_t modp_words_per_chunk =
        static_cast<uint32_t>(std::ceil(static_cast<double>(bytes_per_chunk) * 8.0 / LOGP));
    assert(modp_words_per_chunk <= N);

    // Barrett constants for CRT reduction
    static constexpr uint64_t cr1_0 = 68736257792ULL;   // for MODULUS_0
    static constexpr uint64_t cr1_1 = 73916747789ULL;   // for MODULUS_1

    // Output flat buffer: instances * trials * N * num_per * dim0 CRT-packed u64s
    const size_t total_words = static_cast<size_t>(instances) * trials * N * num_per * dim0;
    std::vector<uint64_t> host_buf(total_words, 0ULL);

    // Working buffers for one DB item
    std::vector<uint64_t> poly_raw(N, 0ULL);
    std::vector<uint64_t> ntt_buf_0(N, 0ULL);
    std::vector<uint64_t> ntt_buf_1(N, 0ULL);

    // Build host NTT tables (singleton — built once)
    get_host_ntt_tables();

    for (uint32_t instance = 0; instance < instances; ++instance) {
        for (uint32_t trial = 0; trial < trials; ++trial) {
            for (uint32_t i = 0; i < num_items; ++i) {
                const uint32_t ii = i % num_per;
                const uint32_t j  = i / num_per;

                // Compute byte offset of this chunk in the file
                // idx_item_in_file = i * db_item_size
                // idx_chunk = instance * trials + trial
                // idx_poly_in_file = idx_item_in_file + idx_chunk * bytes_per_chunk
                const size_t idx_item   = static_cast<size_t>(i) * p.db_item_size;
                const size_t idx_chunk  = instance * trials + trial;
                const size_t byte_off   = idx_item + idx_chunk * bytes_per_chunk;

                // Zero the polynomial
                std::fill(poly_raw.begin(), poly_raw.end(), 0ULL);

                // How many bytes are available.
                // LOGP=8: each coefficient is exactly one byte, so we can read
                // directly without the 8-byte-aligned read_arbitrary_bits, which
                // would otherwise read past the end of the mmap'd region.
                uint32_t words_read = 0;
                if (byte_off < file_size) {
                    const size_t avail      = file_size - byte_off;
                    const size_t bytes_read = std::min(static_cast<size_t>(bytes_per_chunk), avail);
                    words_read = static_cast<uint32_t>(bytes_read);  // LOGP=8: 1 byte per word
                    if (words_read > N) words_read = N;

                    const uint8_t* chunk = raw_tiles + byte_off;
                    for (uint32_t z = 0; z < words_read; ++z) {
                        uint64_t val = chunk[z];  // LOGP=8: direct byte read, no alignment issue
                        // recenter: map [0, P-1] into signed range, then lift to [0, modulus)
                        poly_raw[z] = recenter_mod(val, SpiralParams::P, LARGE_MOD);
                    }
                }

                // CRT-reduce each coefficient
                for (uint32_t z = 0; z < N; ++z) {
                    ntt_buf_0[z] = host_barrett(poly_raw[z], SpiralParams::MODULUS_0, cr1_0);
                    ntt_buf_1[z] = host_barrett(poly_raw[z], SpiralParams::MODULUS_1, cr1_1);
                }

                // NTT forward on each CRT slot
                host_ntt_fwd(ntt_buf_0.data(), 0);
                host_ntt_fwd(ntt_buf_1.data(), 1);

                // CRT-pack and store at calc_index order
                // calc_index([instance, trial, z, ii, j],
                //             [instances, trials, N, num_per, dim0])
                // = instance*(trials*N*num_per*dim0)
                //   + trial*(N*num_per*dim0)
                //   + z*(num_per*dim0)
                //   + ii*dim0
                //   + j
                const size_t base_inst  = static_cast<size_t>(instance) * trials * N * num_per * dim0;
                const size_t base_trial = static_cast<size_t>(trial) * N * num_per * dim0;

                for (uint32_t z = 0; z < N; ++z) {
                    const size_t idx_dst = base_inst + base_trial
                        + static_cast<size_t>(z) * num_per * dim0
                        + static_cast<size_t>(ii) * dim0
                        + j;
                    host_buf[idx_dst] = (ntt_buf_0[z] & 0xFFFFFFFFULL) | (ntt_buf_1[z] << 32);
                }
            }
        }
    }

    // Allocate and upload to GPU
    DeviceDB db;
    db.num_words = total_words;
    cudaError_t err = cudaMalloc(&db.d_data, total_words * sizeof(uint64_t));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("load_db_to_gpu: cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaMemcpy(db.d_data, host_buf.data(), total_words * sizeof(uint64_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(db.d_data);
        db.d_data = nullptr;
        throw std::runtime_error(std::string("load_db_to_gpu: cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }

    return db;
}
