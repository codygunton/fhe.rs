#pragma once
// GPU matrix types and data structures for Spiral PIR.
//
// Memory layout (modulus-major, matches spiral-rs PolyMatrixNTT):
//   For a rows×cols polynomial matrix, the flat buffer has
//   rows * cols * CRT_COUNT * POLY_LEN  u64 words.
//   Index: data[row*cols*CRT_COUNT*POLY_LEN + col*CRT_COUNT*POLY_LEN + crt*POLY_LEN + coeff]
//
// CRT-packed u64 (used in reoriented query buffers and the database):
//   packed = (mod0_coeff & 0xFFFFFFFF) | (mod1_coeff << 32)

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "params.hpp"

// ── DevicePolyMatrix ──────────────────────────────────────────────────────────
// Owns a device-side buffer holding a rows×cols polynomial matrix.
// Buffer size: rows * cols * CRT_COUNT * POLY_LEN * sizeof(uint64_t) bytes.
//
// Non-owning views can be created with DevicePolyMatrix::view() — these hold
// a raw pointer into an externally-managed buffer and do NOT free it on destruction.
struct DevicePolyMatrix {
    uint64_t* d_data = nullptr;
    uint32_t  rows   = 0;
    uint32_t  cols   = 0;
    bool      owned_ = true;  // false for non-owning views (see view())

    DevicePolyMatrix() = default;
    DevicePolyMatrix(uint32_t r, uint32_t c);
    ~DevicePolyMatrix();

    // Non-copyable: avoids accidental GPU allocation duplication.
    DevicePolyMatrix(const DevicePolyMatrix&)            = delete;
    DevicePolyMatrix& operator=(const DevicePolyMatrix&) = delete;

    // Move-constructible so we can store in std::vector.
    // Propagates owned_ so views stay non-owning after move.
    DevicePolyMatrix(DevicePolyMatrix&& other) noexcept;
    DevicePolyMatrix& operator=(DevicePolyMatrix&& other) noexcept;

    // Create a non-owning view into an externally-managed device buffer.
    // The caller must ensure the buffer outlives the returned view.
    static DevicePolyMatrix view(uint64_t* ptr, uint32_t r, uint32_t c) {
        DevicePolyMatrix m;
        m.d_data = ptr; m.rows = r; m.cols = c; m.owned_ = false;
        return m;
    }

    // Number of u64 words in this matrix.
    size_t num_words() const {
        return SpiralParams::poly_words(rows, cols);
    }

    // Byte size of the device buffer.
    size_t byte_size() const {
        return num_words() * sizeof(uint64_t);
    }

    // Copy host data → device.
    void upload(const uint64_t* host_data, size_t num_words_to_copy = 0,
                cudaStream_t stream = 0);

    // Copy device data → host buffer (must be pre-allocated).
    void download(uint64_t* host_data, size_t num_words_to_copy = 0,
                  cudaStream_t stream = 0) const;

    bool valid() const { return d_data != nullptr; }
};

// ── CiphertextGPU ─────────────────────────────────────────────────────────────
// A Spiral RLWE ciphertext: a 2×1 polynomial matrix in NTT domain.
struct CiphertextGPU {
    DevicePolyMatrix poly;  // rows=2, cols=1

    CiphertextGPU() : poly(2, 1) {}
};

// ── PublicParamsGPU ───────────────────────────────────────────────────────────
// Deserialized and GPU-uploaded Spiral public parameters (from /api/setup).
//
// Mirrors spiral-rs PublicParameters with expand_queries=true, version=0.
//
// v_packing:          N matrices, each (N+1)×T_CONV
// v_expansion_left:   g() matrices, each 2×T_EXP_LEFT
// v_expansion_right:  (stop_round()+1) matrices, each 2×T_EXP_RIGHT
// v_conversion:       1 matrix, 2×(2*T_CONV)
struct PublicParamsGPU {
    std::vector<DevicePolyMatrix> v_packing;
    std::vector<DevicePolyMatrix> v_expansion_left;
    std::vector<DevicePolyMatrix> v_expansion_right;
    std::vector<DevicePolyMatrix> v_conversion;

    PublicParamsGPU() = default;
    // Non-copyable (DevicePolyMatrix is non-copyable).
    PublicParamsGPU(const PublicParamsGPU&)            = delete;
    PublicParamsGPU& operator=(const PublicParamsGPU&) = delete;
    PublicParamsGPU(PublicParamsGPU&&)                 = default;
    PublicParamsGPU& operator=(PublicParamsGPU&&)      = default;
};

// ── DeviceDB ─────────────────────────────────────────────────────────────────
// The preprocessed Spiral database in GPU VRAM.
//
// Layout mirrors spiral-rs load_db_from_seek() calc_index ordering:
//   flat index = calc_index([instance, trial, z, ii, j],
//                            [instances, trials, poly_len, num_per, dim0])
//   = instance*trials*poly_len*num_per*dim0
//     + trial*poly_len*num_per*dim0
//     + z*num_per*dim0
//     + ii*dim0
//     + j
//
// Each word is a CRT-packed u64: (ntt_mod0 & 0xFFFFFFFF) | (ntt_mod1 << 32)
struct DeviceDB {
    uint64_t* d_data    = nullptr;
    size_t    num_words = 0;  // = instances * trials * poly_len * num_per * dim0

    DeviceDB() = default;
    ~DeviceDB();

    DeviceDB(const DeviceDB&)            = delete;
    DeviceDB& operator=(const DeviceDB&) = delete;
    DeviceDB(DeviceDB&& other) noexcept;
    DeviceDB& operator=(DeviceDB&& other) noexcept;

    bool valid() const { return d_data != nullptr; }

    // Byte size of the device buffer.
    size_t byte_size() const { return num_words * sizeof(uint64_t); }
};
