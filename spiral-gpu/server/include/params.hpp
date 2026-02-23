#pragma once
// SpiralParams — compile-time and runtime parameters for Spiral PIR.
//
// This is a direct C++ port of spiral-rs src/params.rs.  All derived formulas
// (setup_bytes, query_bytes, response_bytes, g, stop_round, …) are kept
// byte-identical to the Rust originals so that the same WASM client works with
// this GPU server unchanged.
//
// Fixed parameters (compile-time constants) match the production configuration:
//   poly_len=2048, moduli=[268369921, 249561089], t_conv=4, t_exp_left=5,
//   t_exp_right=56, t_gsw=7, n=2, p=256, q2_bits=22

#include <cstddef>
#include <cstdint>
#include <string>

// ── Constant memory layout ────────────────────────────────────────────────────
// NTT polynomial layout mirrors spiral-rs (modulus-major):
//   data[crt * POLY_LEN + coeff]   for a single (1×1) polynomial
//   data[row*cols*CRT_COUNT*POLY_LEN + col*CRT_COUNT*POLY_LEN + crt*POLY_LEN + coeff]
//   for a matrix polynomial
//
// CRT-packed u64 (used in reoriented query buffers and DB):
//   packed = (mod0_val & 0xFFFFFFFF) | (mod1_val << 32)

static constexpr uint32_t SPIRAL_POLY_LEN  = 2048;
static constexpr uint32_t SPIRAL_CRT_COUNT = 2;
static constexpr uint64_t SPIRAL_MODULUS_0 = 268369921ULL;   // q0, 28-bit prime
static constexpr uint64_t SPIRAL_MODULUS_1 = 249561089ULL;   // q1, 28-bit prime
static constexpr uint32_t SPIRAL_SEED_LEN  = 32;             // ChaCha20 seed bytes

// Q2 table from spiral-rs params.rs (indexed by q2_bits)
// Q2_VALUES[22] = 3604481
static constexpr uint64_t SPIRAL_Q2_VALUES[37] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    12289,12289,61441,65537,65537,520193,786433,786433,
    3604481,7340033,16515073,33292289,67043329,132120577,
    268369921,469762049,1073479681,2013265921,4293918721ULL,
    8588886017ULL,17175674881ULL,34359214081ULL,68718428161ULL,
};

// ── SpiralParams ──────────────────────────────────────────────────────────────
struct SpiralParams {
    // ── Fixed compile-time constants ─────────────────────────────────────────
    static constexpr uint32_t POLY_LEN    = SPIRAL_POLY_LEN;
    static constexpr uint32_t CRT_COUNT   = SPIRAL_CRT_COUNT;
    static constexpr uint64_t MODULUS_0   = SPIRAL_MODULUS_0;
    static constexpr uint64_t MODULUS_1   = SPIRAL_MODULUS_1;
    static constexpr uint32_t N           = 2;      // matrix dimension n×n
    static constexpr uint32_t P           = 256;    // plaintext modulus
    static constexpr uint32_t Q2_BITS     = 22;
    static constexpr uint32_t T_CONV      = 4;
    static constexpr uint32_t T_EXP_LEFT  = 5;
    static constexpr uint32_t T_EXP_RIGHT = 56;
    static constexpr uint32_t T_GSW       = 7;

    // ── Runtime parameters (set from CLI / tile-mapping JSON) ────────────────
    uint32_t nu_1;          // db_dim_1 exponent; default 9 → 512 first-dim entries
    uint32_t nu_2;          // db_dim_2 exponent; default 4 → 16 second-dim entries
    uint32_t instances;     // default 3
    uint32_t db_item_size;  // bytes per item in the flat tile file; default 20480

    // ── Constructor: production defaults ─────────────────────────────────────
    SpiralParams()
        : nu_1(9), nu_2(4), instances(3), db_item_size(20480) {}

    SpiralParams(uint32_t nu_1_, uint32_t nu_2_, uint32_t instances_, uint32_t db_item_size_)
        : nu_1(nu_1_), nu_2(nu_2_), instances(instances_), db_item_size(db_item_size_) {}

    // ── Derived dimensions ────────────────────────────────────────────────────

    // 2^nu_1 = number of first-dimension entries (dim0)
    uint32_t dim0() const { return 1u << nu_1; }

    // 2^nu_2 = number of second-dimension entries (num_per)
    uint32_t num_per() const { return 1u << nu_2; }

    // Total number of database items
    uint32_t num_items() const { return dim0() * num_per(); }

    // g() — matches spiral-rs Params::g()
    // Number of expansion levels: ceil(log2(t_gsw * nu_2 + dim0))
    uint32_t g() const {
        uint32_t num_bits_to_gen = T_GSW * nu_2 + dim0();
        return log2_ceil(num_bits_to_gen);
    }

    // stop_round() — matches spiral-rs Params::stop_round()
    // Last level at which right-expansion keys are used: ceil(log2(t_gsw * nu_2))
    uint32_t stop_round() const {
        return log2_ceil(T_GSW * nu_2);
    }

    // ── Size queries (match spiral-rs Params byte-for-byte) ──────────────────

    // setup_bytes() — matches spiral-rs Params::setup_bytes() with version=0,
    // expand_queries=true, t_exp_left != t_exp_right.
    //
    // Serialized layout:
    //   [SEED_LEN bytes seed]
    //   [N packing matrices, each (N+1-1)*T_CONV = N*T_CONV polys of POLY_LEN u64s]
    //   [g() expansion-left matrices, each 1 * T_EXP_LEFT polys]
    //   [(stop_round()+1) expansion-right matrices, each 1 * T_EXP_RIGHT polys]
    //   [1 conversion matrix, 1 * 2*T_CONV polys]
    //
    // Each polynomial is POLY_LEN u64s.  The first row of each matrix is
    // regenerated from the RNG seed and is NOT serialized (serialize_polymatrix_for_rng
    // skips the first row, hence packing_sz = (rows-1)*t_conv).
    size_t setup_bytes() const {
        uint32_t sz_polys = 0;

        // Packing matrices: N matrices of size (N+1)×T_CONV, serialize rows-1=N rows
        sz_polys += N * (N * T_CONV);

        // Expansion left: g() matrices of size 2×T_EXP_LEFT, serialize rows-1=1 row
        sz_polys += g() * T_EXP_LEFT;

        // Expansion right: (stop_round()+1) matrices of size 2×T_EXP_RIGHT
        sz_polys += (stop_round() + 1) * T_EXP_RIGHT;

        // Conversion: 1 matrix of size 2×(2*T_CONV), serialize rows-1=1 row
        sz_polys += 2 * T_CONV;

        return SPIRAL_SEED_LEN + sz_polys * POLY_LEN * sizeof(uint64_t);
    }

    // query_bytes() — matches spiral-rs Params::query_bytes() with expand_queries=true
    //
    // Serialized layout:
    //   [SEED_LEN bytes seed]
    //   [1 query polynomial: 1 row of the 2×1 query matrix, POLY_LEN u64s]
    size_t query_bytes() const {
        // expand_queries=true: one 2×1 raw ciphertext, serialize rows-1=1 polynomial
        return SPIRAL_SEED_LEN + POLY_LEN * sizeof(uint64_t);
    }

    // response_bytes() — matches spiral-rs server.rs encode() output length
    //
    // For each instance:
    //   N * POLY_LEN coefficients at q2_bits each  (first row of packed ct)
    //   N*N * POLY_LEN coefficients at q1_bits each (rest rows)
    // q1 = 4 * P, q1_bits = ceil(log2(4*P)) = ceil(log2(1024)) = 10
    // q2 = Q2_VALUES[Q2_BITS] = 3604481, q2_bits = Q2_BITS = 22
    // Rounded up to next multiple of 8 bytes (64 bits).
    size_t response_bytes() const {
        // q1 = 4 * P = 1024, q1_bits = ceil(log2(1024)) = 10
        constexpr uint32_t q1_bits  = 10;
        constexpr uint32_t q2_bits  = Q2_BITS;         // 22

        uint32_t num_bits = instances
            * (q2_bits  * N     * POLY_LEN   // first row
            +  q1_bits  * N * N * POLY_LEN); // rest rows

        constexpr uint32_t round_to = 64;
        return ((num_bits + round_to - 1) / round_to) * round_to / 8;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Number of u64 words for a matrix polynomial of dimensions rows×cols.
    // Layout: rows * cols * CRT_COUNT * POLY_LEN u64s.
    static size_t poly_words(uint32_t rows, uint32_t cols) {
        return (size_t)rows * cols * CRT_COUNT * POLY_LEN;
    }

    // Produce the JSON string that the spiral-wasm WASM client expects
    // (/api/params response field "spiral_params").
    std::string to_json() const;

    // ── Internal ──────────────────────────────────────────────────────────────
private:
    // ceil(log2(n)), matching spiral-rs number_theory::log2_ceil_usize.
    // Returns 0 for n <= 1.
    static uint32_t log2_ceil(uint32_t n) {
        if (n <= 1) return 0;
        return 32u - static_cast<uint32_t>(__builtin_clz(n - 1));
    }
};

// select_params() — mirrors spiral-cpu/server select_params_json() exactly.
//
// Chooses nu_2 and instances so the database can hold num_tiles items each of
// tile_size bytes.  nu_1 is fixed at 9 (matches Blyss v1 production shape).
// Uses t_conv=4, t_exp_right=56 (correct noise-free parameters).
SpiralParams select_params(size_t num_tiles, size_t tile_size);
