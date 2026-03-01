#pragma once
// Host-side NTT tables and forward/inverse NTT for Spiral PIR preprocessing.
//
// Used by serialization.cpp, database.cu, and expand_query.cu to
// CRT-reduce and NTT-transform polynomials before uploading to GPU.
//
// Mirrors spiral-rs src/ntt.rs (non-AVX2 path).
// Layout: each "slot" is POLY_LEN u64 values for one CRT modulus.

#include "params.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

// ── Helper functions ──────────────────────────────────────────────────────────

static inline uint64_t hntt_mod_inv(uint64_t value, uint64_t modulus) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)modulus, newr = (int64_t)value;
    while (newr != 0) {
        int64_t q = r / newr, tmp;
        tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (t < 0) t += (int64_t)modulus;
    return (uint64_t)t;
}

static inline uint64_t hntt_mod_exp(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1, power = base % mod;
    while (exp > 0) {
        if (exp & 1)
            result = (uint64_t)(((unsigned __int128)result * power) % mod);
        exp >>= 1;
        power = (uint64_t)(((unsigned __int128)power * power) % mod);
    }
    return result;
}

static inline uint64_t hntt_div2(uint64_t v, uint64_t mod) {
    if (v & 1) {
        unsigned __int128 s = (unsigned __int128)v + mod;
        return (uint64_t)(s >> 1);
    }
    return v >> 1;
}

static inline uint64_t hntt_rev_bits(uint64_t x, int bits) {
    uint64_t r = 0;
    for (int i = 0; i < 64; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r >> (64 - bits);
}

static inline uint64_t hntt_find_root(uint64_t poly_len, uint64_t mod) {
    uint64_t degree = 2 * poly_len;
    uint64_t size_q = (mod - 1) / degree;
    for (uint64_t g = 2; g < mod; ++g) {
        uint64_t root = hntt_mod_exp(g, size_q, mod);
        if (hntt_mod_exp(root, degree / 2, mod) == mod - 1) {
            uint64_t gen_sq = (uint64_t)(((unsigned __int128)root * root) % mod);
            uint64_t cur = root, minimal = root;
            for (uint64_t i = 0; i < degree; ++i) {
                if (cur < minimal) minimal = cur;
                cur = (uint64_t)(((unsigned __int128)cur * gen_sq) % mod);
            }
            return minimal;
        }
    }
    throw std::runtime_error("hntt_find_root: no root found");
}

// ── NTT tables ────────────────────────────────────────────────────────────────

struct HostNttTables {
    std::vector<uint64_t> fwd[2], fwdp[2], inv[2], invp[2];
    bool built = false;

    void build() {
        if (built) return;
        const uint32_t N   = SpiralParams::POLY_LEN;
        const int      logn = 11;
        const uint64_t mods[2] = {SpiralParams::MODULUS_0, SpiralParams::MODULUS_1};
        for (int m = 0; m < 2; ++m) {
            uint64_t mod     = mods[m];
            uint32_t mod_u32 = (uint32_t)mod;
            uint64_t root    = hntt_find_root(N, mod);
            uint64_t inv_root = hntt_mod_inv(root, mod);

            fwd[m].resize(N); fwdp[m].resize(N);
            inv[m].resize(N); invp[m].resize(N);

            // Forward twiddles
            uint64_t pw = root;
            fwd[m][0] = 1;
            for (uint32_t i = 1; i < N; ++i) {
                fwd[m][(uint32_t)hntt_rev_bits(i, logn)] = pw;
                pw = (uint64_t)(((unsigned __int128)pw * root) % mod);
            }
            for (uint32_t i = 0; i < N; ++i) {
                uint64_t wv = fwd[m][i] << 32;
                fwdp[m][i] = (uint32_t)(wv / mod_u32);
            }

            // Inverse twiddles
            pw = inv_root;
            inv[m][0] = 1;
            for (uint32_t i = 1; i < N; ++i) {
                inv[m][(uint32_t)hntt_rev_bits(i, logn)] = pw;
                pw = (uint64_t)(((unsigned __int128)pw * inv_root) % mod);
            }
            for (uint32_t i = 0; i < N; ++i) inv[m][i] = hntt_div2(inv[m][i], mod);
            for (uint32_t i = 0; i < N; ++i) {
                uint64_t wv = inv[m][i] << 32;
                invp[m][i] = (uint32_t)(wv / mod_u32);
            }
        }
        built = true;
    }
};

inline HostNttTables& get_host_ntt_tables() {
    static HostNttTables t;
    t.build();
    return t;
}

// Forward NTT in-place on one CRT slot (N = 2048 u64 values).
// crt_idx: 0 = MODULUS_0, 1 = MODULUS_1.
inline void host_ntt_fwd(uint64_t* buf, int crt_idx) {
    HostNttTables& t = get_host_ntt_tables();
    const uint32_t N    = SpiralParams::POLY_LEN;
    const int      logn = 11;
    const auto&    fwd  = t.fwd[crt_idx];
    const auto&    fwdp = t.fwdp[crt_idx];
    const uint64_t mod  = (crt_idx == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;
    const uint32_t q    = (uint32_t)mod;
    const uint32_t twoq = 2 * q;

    for (int mm = 0; mm < logn; ++mm) {
        const uint32_t m = 1u << mm;
        const uint32_t t_half = N >> (mm + 1);
        for (uint32_t i = 0; i < m; ++i) {
            const uint64_t w  = fwd[m + i];
            const uint64_t wp = fwdp[m + i];
            for (uint32_t j = 0; j < t_half; ++j) {
                const uint32_t upper = i * (2 * t_half) + j;
                const uint32_t lower = upper + t_half;
                uint32_t x = (uint32_t)buf[upper];
                uint32_t y = (uint32_t)buf[lower];
                uint32_t cx = x - twoq * (x >= twoq);
                uint64_t qt = ((uint64_t)y * wp) >> 32;
                uint64_t qn = w * (uint64_t)y - qt * (uint64_t)q;
                buf[upper] = (uint64_t)cx + qn;
                buf[lower] = (uint64_t)cx + (uint64_t)twoq - qn;
            }
        }
    }
    for (uint32_t i = 0; i < N; ++i) {
        if (buf[i] >= twoq) buf[i] -= twoq;
        if (buf[i] >= q)    buf[i] -= q;
    }
}

// Inverse NTT in-place on one CRT slot.
inline void host_ntt_inv(uint64_t* buf, int crt_idx) {
    HostNttTables& t = get_host_ntt_tables();
    const uint32_t N    = SpiralParams::POLY_LEN;
    const int      logn = 11;
    const auto&    inv  = t.inv[crt_idx];
    const auto&    invp = t.invp[crt_idx];
    const uint64_t mod  = (crt_idx == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;
    const uint64_t twoq = 2 * mod;

    for (int mm = logn - 1; mm >= 0; --mm) {
        const uint32_t h = 1u << mm;
        const uint32_t t_half = N >> (mm + 1);
        for (uint32_t i = 0; i < h; ++i) {
            const uint64_t w  = inv[h + i];
            const uint64_t wp = invp[h + i];
            for (uint32_t j = 0; j < t_half; ++j) {
                const uint32_t upper = i * (2 * t_half) + j;
                const uint32_t lower = upper + t_half;
                uint64_t x = buf[upper], y = buf[lower];
                uint64_t t_tmp  = twoq - y + x;
                uint64_t curr_x = x + y - twoq * ((x << 1) >= t_tmp ? 1 : 0);
                uint64_t ht     = (t_tmp * wp) >> 32;
                uint64_t res_x  = (curr_x + mod * (t_tmp & 1)) >> 1;
                uint64_t res_y  = w * t_tmp - ht * mod;
                buf[upper] = res_x;
                buf[lower] = res_y;
            }
        }
    }
    for (uint32_t i = 0; i < N; ++i) {
        if (buf[i] >= twoq) buf[i] -= twoq;
        if (buf[i] >= mod)  buf[i] -= mod;
    }
}

// Barrett reduction mod qi (small modulus, cr1 precomputed).
static inline uint64_t host_barrett_reduce(uint64_t val, uint64_t mod, uint64_t cr1) {
    uint64_t q = (uint64_t)(((unsigned __int128)val * cr1) >> 64);
    uint64_t r = val - q * mod;
    if (r >= mod) r -= mod;
    return r;
}

// CRT-reduce a raw polynomial (rows*cols*POLY_LEN source, modulus ≤ LARGE_MOD)
// and NTT-transform it into the NTT-domain buffer (rows*cols*CRT_COUNT*POLY_LEN dest).
inline void host_to_ntt(const uint64_t* raw, uint64_t* ntt_out,
                         uint32_t rows, uint32_t cols) {
    const uint32_t N   = SpiralParams::POLY_LEN;
    const uint32_t CRT = SpiralParams::CRT_COUNT;
    // cr1 values for Barrett mod q0 and q1
    static constexpr uint64_t cr1_0 = 68736257792ULL;
    static constexpr uint64_t cr1_1 = 73916747789ULL;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint64_t* src = raw     + (r * cols + c) * N;
            uint64_t*       dst = ntt_out + (r * cols + c) * CRT * N;

            for (uint32_t z = 0; z < N; ++z) {
                dst[0 * N + z] = host_barrett_reduce(src[z], SpiralParams::MODULUS_0, cr1_0);
                dst[1 * N + z] = host_barrett_reduce(src[z], SpiralParams::MODULUS_1, cr1_1);
            }
            host_ntt_fwd(dst + 0 * N, 0);
            host_ntt_fwd(dst + 1 * N, 1);
        }
    }
}
