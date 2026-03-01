// Spiral binary format: parse PublicParameters + Query, encode response.
//
// Mirrors spiral-rs client.rs PublicParameters::deserialize() / Query::deserialize()
// and server.rs encode().
//
// All u64s in the binary format are native-endian (little-endian on x86_64).
// ChaCha20 RNG matches rand_chacha::ChaCha20Rng::from_seed (IETF variant:
//   96-bit nonce, 32-bit counter, 20 rounds).

#include "serialization.hpp"
#include "params.hpp"
#include "types.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cmath>

// These are defined in ntt.cu — we use them after init_ntt_tables() has run.
void launch_ntt_forward(uint64_t* d_polys, uint32_t count, cudaStream_t stream);
void launch_ntt_inverse(uint64_t* d_polys, uint32_t count, cudaStream_t stream);

#include <cuda_runtime.h>

// ── Compile-time constants derived from the two Spiral moduli ─────────────────

// Large modulus = q0 * q1  (fits in u64: ~6.7e16 < 2^56)
static constexpr uint64_t LARGE_MOD =
    SpiralParams::MODULUS_0 * SpiralParams::MODULUS_1;
// 268369921 * 249561089 = 66974689739603969

// Barrett constants for LARGE_MOD reduction (cr0 from spiral-rs arith.rs tests):
//   get_barrett_crs(66974689739603969) == (7906011006380390721, 275)
static constexpr uint64_t LARGE_MOD_CR0 = 7906011006380390721ULL;
static constexpr uint64_t LARGE_MOD_CR1 = 275ULL;

// CRT composition coefficients:
//   mod0_inv_mod1 = q0 * invert(q0, q1) mod (q0*q1)
//   mod1_inv_mod0 = q1 * invert(q1, q0) mod (q0*q1)
// Computed once via extended GCD; stored as precomputed constants for speed.
// These are computed at first use in crt_compose().
static uint64_t g_mod0_inv_mod1 = 0;
static uint64_t g_mod1_inv_mod0 = 0;

// ── ChaCha20 RNG (IETF variant, 20 rounds) ────────────────────────────────────
// Matches rand_chacha::ChaCha20Rng::from_seed output byte-for-byte.

static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

#define QUARTERROUND(a, b, c, d)    \
    a += b; d ^= a; d = rotl32(d, 16); \
    c += d; b ^= c; b = rotl32(b, 12); \
    a += b; d ^= a; d = rotl32(d, 8);  \
    c += d; b ^= c; b = rotl32(b, 7)

static void chacha20_block(uint32_t out[16], const uint32_t in[16]) {
    uint32_t x[16];
    memcpy(x, in, 64);
    for (int i = 0; i < 10; ++i) {
        QUARTERROUND(x[0], x[4], x[8],  x[12]);
        QUARTERROUND(x[1], x[5], x[9],  x[13]);
        QUARTERROUND(x[2], x[6], x[10], x[14]);
        QUARTERROUND(x[3], x[7], x[11], x[15]);
        QUARTERROUND(x[0], x[5], x[10], x[15]);
        QUARTERROUND(x[1], x[6], x[11], x[12]);
        QUARTERROUND(x[2], x[7], x[8],  x[13]);
        QUARTERROUND(x[3], x[4], x[9],  x[14]);
    }
    for (int i = 0; i < 16; ++i) out[i] = x[i] + in[i];
}

static inline uint32_t le_load32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t le_load64(const uint8_t* p) {
    return (uint64_t)le_load32(p) | ((uint64_t)le_load32(p + 4) << 32);
}

struct ChaCha20Rng {
    uint32_t state[16];   // current key schedule state
    uint8_t  block[64];   // current block
    int      pos;         // byte position in block (64 = exhausted)

    void init(const uint8_t seed[32]) {
        // Constant words "expand 32-byte k"
        state[0] = 0x61707865u;
        state[1] = 0x3320646eu;
        state[2] = 0x79622d32u;
        state[3] = 0x6b206574u;
        // Key
        for (int i = 0; i < 8; ++i)
            state[4 + i] = le_load32(seed + 4 * i);
        // Counter = 0, nonce = 0
        state[12] = 0; state[13] = 0; state[14] = 0; state[15] = 0;
        pos = 64;  // force refill on first call
    }

    void refill() {
        uint32_t out[16];
        chacha20_block(out, state);
        // Write output as LE bytes
        for (int i = 0; i < 16; ++i) {
            block[4*i + 0] = (uint8_t)(out[i]       );
            block[4*i + 1] = (uint8_t)(out[i] >> 8  );
            block[4*i + 2] = (uint8_t)(out[i] >> 16 );
            block[4*i + 3] = (uint8_t)(out[i] >> 24 );
        }
        state[12]++;  // increment 32-bit block counter
        pos = 0;
    }

    // Generate one u64 (8 bytes LE from keystream) — mirrors rng.gen::<u64>().
    uint64_t gen_u64() {
        if (pos + 8 > 64) refill();
        uint64_t v = le_load64(block + pos);
        pos += 8;
        return v;
    }
};

// ── Modular arithmetic helpers ─────────────────────────────────────────────────

static uint64_t host_mod_inv_u64(uint64_t value, uint64_t modulus) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)modulus, newr = (int64_t)value;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp;
        tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (t < 0) t += (int64_t)modulus;
    return (uint64_t)t;
}

// Barrett reduction for u128 mod LARGE_MOD.
// Mirrors barrett_raw_u128 from spiral-rs arith.rs.
static uint64_t barrett_u128_large(unsigned __int128 val) {
    uint64_t zx = (uint64_t)(val & ((unsigned __int128)0xFFFFFFFFFFFFFFFFULL));
    uint64_t zy = (uint64_t)(val >> 64);

    // Mirrors the Rust barrett_raw_u128 implementation using cr0, cr1:
    //   (zx, zy) are the lo/hi halves of val
    //   Following the Rust code verbatim with u64 carry arithmetic
    uint64_t tmp1 = 0, tmp3 = 0, carry = 0;

    // prod = mul_u128(zx, cr0) → we only need hi half
    unsigned __int128 prod0 = (unsigned __int128)zx * LARGE_MOD_CR0;
    carry = (uint64_t)(prod0 >> 64);

    // tmp2 = mul_u128(zx, cr1)
    unsigned __int128 prod1 = (unsigned __int128)zx * LARGE_MOD_CR1;
    uint64_t tmp2x = (uint64_t)(prod1 & 0xFFFFFFFFFFFFFFFFULL);
    uint64_t tmp2y = (uint64_t)(prod1 >> 64);
    // tmp3 = tmp2y + add_u64(tmp2x, carry, &tmp1)
    uint64_t add_res = tmp2x + carry;
    tmp1 = add_res;
    uint64_t add_carry = (add_res < tmp2x) ? 1 : 0;
    tmp3 = tmp2y + add_carry;

    // tmp2 = mul_u128(zy, cr0)
    unsigned __int128 prod2 = (unsigned __int128)zy * LARGE_MOD_CR0;
    tmp2x = (uint64_t)(prod2 & 0xFFFFFFFFFFFFFFFFULL);
    tmp2y = (uint64_t)(prod2 >> 64);
    // carry = tmp2y + add_u64(tmp1, tmp2x, &tmp1)
    uint64_t add_res2 = tmp1 + tmp2x;
    tmp1 = add_res2;
    uint64_t add_carry2 = (add_res2 < tmp1) ? 1 : 0;  // intentional: tmp1 was just updated
    // Actually re-read: add_u64(tmp1, tmp2x, &tmp1) → tmp1 = tmp1+tmp2x, return (tmp1_new < tmp1_old)
    // But tmp1 was updated — we need the pre-update value
    // Let me redo this more carefully:
    {
        uint64_t old_tmp1 = tmp1;
        // oh wait, tmp1 was just set to add_res2 above
        // Let me re-implement exactly as in the Rust:
        //   carry = tmp2y + add_u64(tmp1_current_before_this_op, tmp2x, &tmp1)
        // The Rust code does:
        //   (tmp2x, tmp2y) = mul_u128(zy, cr0)
        //   carry = tmp2y + add_u64(tmp1, tmp2x, &tmp1);  <- this uses the CURRENT tmp1
        // Since I just set tmp1 = add_res2 (= old_tmp1 + tmp2x), and add_carry2 = borrow:
        // add_carry2 should be (add_res2 < old_tmp1_before_this_assignment)
        // But I reassigned tmp1 above. Let me fix:
        (void)add_carry2;  // discard the wrong carry
        // The correct carry:
        add_carry2 = (old_tmp1 != add_res2) ? 0 : 0;  // dummy, recalculate below
        (void)add_carry2;
        // Actually let's just use an intermediate
        uint64_t old_t = tmp3;  // save tmp3 from above
        // Recompute from scratch what Rust does:
        //   (tmp2x, tmp2y) = mul_u128(zy, cr0)
        //   carry = tmp2y + add_u64(tmp1_saved, tmp2x, &tmp1)
        // where tmp1_saved is the value of tmp1 BEFORE this assignment
        // tmp1_saved was set in the first mul_u128(zx,cr1) step
        // I need to not clobber it. Let me redo the whole thing.
        (void)old_t;
    }
    // Simpler: just compute directly
    {
        uint64_t t1 = 0, t3 = 0;
        // From Rust (verbatim):
        //   let (_, prody) = mul_u128(zx, cr0);  → carry = prody = hi(zx*cr0)
        uint64_t carry_val = (uint64_t)(((unsigned __int128)zx * LARGE_MOD_CR0) >> 64);

        //   let (mut tmp2x, mut tmp2y) = mul_u128(zx, cr1);
        unsigned __int128 xc1 = (unsigned __int128)zx * LARGE_MOD_CR1;
        uint64_t t2x = (uint64_t)(xc1);
        uint64_t t2y = (uint64_t)(xc1 >> 64);
        //   tmp3 = tmp2y + add_u64(tmp2x, carry, &tmp1);
        uint64_t sum_xc = t2x + carry_val;
        t1 = sum_xc;
        uint64_t oc = (sum_xc < t2x) ? 1 : 0;
        t3 = t2y + oc;

        //   (tmp2x, tmp2y) = mul_u128(zy, cr0);
        unsigned __int128 yc0 = (unsigned __int128)zy * LARGE_MOD_CR0;
        t2x = (uint64_t)(yc0);
        t2y = (uint64_t)(yc0 >> 64);
        //   carry = tmp2y + add_u64(tmp1, tmp2x, &tmp1);
        uint64_t old_t1 = t1;
        t1 = t1 + t2x;
        uint64_t oc2 = (t1 < old_t1) ? 1 : 0;
        carry_val = t2y + oc2;

        //   tmp1 = zy * cr1 + tmp3 + carry;
        t1 = zy * LARGE_MOD_CR1 + t3 + carry_val;

        //   tmp3 = zx.wrapping_sub(tmp1.wrapping_mul(modulus));
        t3 = zx - t1 * LARGE_MOD;  // wrapping arithmetic

        // One correction step (from barrett_reduction_u128_raw):
        t3 -= LARGE_MOD * (t3 >= LARGE_MOD ? 1 : 0);

        (void)tmp1; (void)tmp3; (void)carry;  // suppress warnings on outer vars
        return t3;
    }
}

// Reduce coefficient mod q0 or q1 (Barrett, matching barrett_coeff_u64 in spiral-rs).
static inline uint64_t reduce_mod(uint64_t val, uint64_t modulus, uint64_t cr1) {
    uint64_t q = (uint64_t)(((unsigned __int128)val * cr1) >> 64);
    uint64_t r = val - q * modulus;
    if (r >= modulus) r -= modulus;
    return r;
}

// Rescale a coefficient from inp_mod to out_mod.
// Mirrors spiral-rs arith.rs rescale().
static uint64_t rescale_coeff(uint64_t a, uint64_t inp_mod, uint64_t out_mod) {
    int64_t  inp_mod_i = (int64_t)inp_mod;
    __int128 out_mod_i = (__int128)out_mod;
    int64_t  inp_val   = (int64_t)(a % inp_mod);
    if (inp_val >= inp_mod_i / 2) inp_val -= inp_mod_i;
    int64_t sign = (inp_val >= 0) ? 1 : -1;
    __int128 val = (__int128)inp_val * (__int128)out_mod;
    __int128 result = (val + (__int128)(sign * (inp_mod_i / 2))) / (__int128)inp_mod;
    result = (result + (__int128)((inp_mod / out_mod) * out_mod) + 2 * out_mod_i) % out_mod_i;
    if (result < 0) result += out_mod_i;
    return (uint64_t)((result + out_mod_i) % out_mod_i);
}

// ── write_arbitrary_bits: mirrors spiral-rs util.rs write_arbitrary_bits ──────

static void write_arbitrary_bits(uint8_t* data, uint64_t val,
                                  size_t bit_offs, size_t num_bits) {
    size_t word_off = bit_offs / 64;
    size_t bit_off_within_word = bit_offs % 64;
    val = val & (num_bits < 64 ? ((1ULL << num_bits) - 1ULL) : ~0ULL);

    if (bit_off_within_word + num_bits <= 64) {
        uint8_t* p = data + word_off * 8;
        uint64_t cur;
        memcpy(&cur, p, 8);
        uint64_t mask = (num_bits < 64 ? ((1ULL << num_bits) - 1ULL) : ~0ULL);
        cur &= ~(mask << bit_off_within_word);
        cur |= val << bit_off_within_word;
        memcpy(p, &cur, 8);
    } else {
        uint8_t* p = data + word_off * 8;
        unsigned __int128 cur;
        memcpy(&cur, p, 16);
        unsigned __int128 mask128 = ((unsigned __int128)1 << num_bits) - 1;
        cur &= ~(mask128 << bit_off_within_word);
        cur |= (unsigned __int128)val << bit_off_within_word;
        memcpy(p, &cur, 16);
    }
}

// ── CRT composition initializer ────────────────────────────────────────────────

static void ensure_crt_constants() {
    if (g_mod0_inv_mod1 != 0) return;
    // mod0_inv_mod1 = q0 * inv(q0, q1)
    uint64_t inv0 = host_mod_inv_u64(SpiralParams::MODULUS_0, SpiralParams::MODULUS_1);
    uint64_t inv1 = host_mod_inv_u64(SpiralParams::MODULUS_1, SpiralParams::MODULUS_0);
    // These can be large: compute mod LARGE_MOD
    g_mod0_inv_mod1 = (uint64_t)(((unsigned __int128)SpiralParams::MODULUS_0 * inv0)
                                  % LARGE_MOD);
    g_mod1_inv_mod0 = (uint64_t)(((unsigned __int128)SpiralParams::MODULUS_1 * inv1)
                                  % LARGE_MOD);
}

// CRT reconstruct from (x mod q0, y mod q1) → value mod (q0*q1).
// Mirrors crt_compose_2.
static uint64_t crt_compose(uint64_t x, uint64_t y) {
    unsigned __int128 val =
          (unsigned __int128)x * g_mod1_inv_mod0
        + (unsigned __int128)y * g_mod0_inv_mod1;
    return barrett_u128_large(val);
}

// ── Host-side CPU NTT (matches ntt.cu algorithm) ──────────────────────────────
// Used to transform deserialized public-parameter polynomials.
// Polynomial layout: modulus-major, buf[0..N) = mod0 coefficients,
//                                   buf[N..2N) = mod1 coefficients.

struct HostNttTables {
    std::vector<uint64_t> fwd[2], fwdp[2], inv[2], invp[2];
    bool initialized = false;
};
static HostNttTables g_host_ntt;

static uint64_t host_mod_exp(uint64_t base, uint64_t exp, uint64_t mod) {
    if (exp == 0) return 1;
    uint64_t result = 1;
    uint64_t power  = base % mod;
    while (exp > 0) {
        if (exp & 1)
            result = (uint64_t)(((unsigned __int128)result * power) % mod);
        exp >>= 1;
        power = (uint64_t)(((unsigned __int128)power * power) % mod);
    }
    return result;
}

static uint64_t host_div2_mod(uint64_t v, uint64_t mod) {
    if (v & 1) {
        unsigned __int128 s = (unsigned __int128)v + mod;
        return (uint64_t)(s >> 1);
    }
    return v >> 1;
}

static uint64_t host_reverse_bits(uint64_t x, int bits) {
    uint64_t r = 0;
    for (int i = 0; i < 64; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r >> (64 - bits);
}

static uint64_t host_find_primitive_root(uint64_t poly_len, uint64_t mod) {
    uint64_t degree = 2 * poly_len;
    uint64_t size_q = (mod - 1) / degree;
    for (uint64_t g = 2; g < mod; ++g) {
        uint64_t root = host_mod_exp(g, size_q, mod);
        if (host_mod_exp(root, degree / 2, mod) == mod - 1) {
            uint64_t gen_sq = (uint64_t)(((unsigned __int128)root * root) % mod);
            uint64_t cur = root, minimal = root;
            for (uint64_t i = 0; i < degree; ++i) {
                if (cur < minimal) minimal = cur;
                cur = (uint64_t)(((unsigned __int128)cur * gen_sq) % mod);
            }
            return minimal;
        }
    }
    throw std::runtime_error("No primitive root found");
}

static void build_host_ntt_tables() {
    if (g_host_ntt.initialized) return;
    const uint32_t N = SpiralParams::POLY_LEN;
    const int logn  = 11;
    const uint64_t mods[2] = { SpiralParams::MODULUS_0, SpiralParams::MODULUS_1 };

    for (int m = 0; m < 2; ++m) {
        uint64_t mod    = mods[m];
        uint32_t mod_u32 = (uint32_t)mod;
        uint64_t root    = host_find_primitive_root(N, mod);
        uint64_t inv_root = host_mod_inv_u64(root, mod);

        auto& fwd  = g_host_ntt.fwd[m];
        auto& fwdp = g_host_ntt.fwdp[m];
        auto& inv  = g_host_ntt.inv[m];
        auto& invp = g_host_ntt.invp[m];

        fwd.resize(N); fwdp.resize(N); inv.resize(N); invp.resize(N);

        // powers_of_primitive_root(root, mod, logn) → bit-reversed positions
        uint64_t pw = root;
        for (uint32_t i = 1; i < N; ++i) {
            uint32_t idx = (uint32_t)host_reverse_bits(i, logn);
            fwd[idx] = pw;
            pw = (uint64_t)(((unsigned __int128)pw * root) % mod);
        }
        fwd[0] = 1;

        // scaled forward table
        for (uint32_t i = 0; i < N; ++i) {
            uint64_t wv = fwd[i] << 32;
            fwdp[i] = (uint32_t)(wv / mod_u32);
        }

        // powers_of_primitive_root(inv_root, ...) + div2
        pw = inv_root;
        for (uint32_t i = 1; i < N; ++i) {
            uint32_t idx = (uint32_t)host_reverse_bits(i, logn);
            inv[idx] = pw;
            pw = (uint64_t)(((unsigned __int128)pw * inv_root) % mod);
        }
        inv[0] = 1;
        for (uint32_t i = 0; i < N; ++i) inv[i] = host_div2_mod(inv[i], mod);

        // scaled inverse table
        for (uint32_t i = 0; i < N; ++i) {
            uint64_t wv = inv[i] << 32;
            invp[i] = (uint32_t)(wv / mod_u32);
        }
    }
    g_host_ntt.initialized = true;
}

// Forward NTT on one polynomial's CRT slot (in-place), matching ntt.rs non-AVX2.
// buf: pointer to N uint64_t values for one CRT modulus slot.
static void host_ntt_forward_1(uint64_t* buf, int crt_idx) {
    const uint32_t N   = SpiralParams::POLY_LEN;
    const int      logn = 11;
    const auto& fwd   = g_host_ntt.fwd[crt_idx];
    const auto& fwdp  = g_host_ntt.fwdp[crt_idx];
    const uint64_t mod  = (crt_idx == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;
    const uint32_t q    = (uint32_t)mod;
    const uint32_t twoq = 2 * q;

    for (int mm = 0; mm < logn; ++mm) {
        const uint32_t m = 1u << mm;
        const uint32_t t = N >> (mm + 1);
        for (uint32_t i = 0; i < m; ++i) {
            const uint64_t w  = fwd[m + i];
            const uint64_t wp = fwdp[m + i];
            for (uint32_t j = 0; j < t; ++j) {
                const uint32_t upper = i * (2 * t) + j;
                const uint32_t lower = upper + t;
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
    // Two-step reduction
    for (uint32_t i = 0; i < N; ++i) {
        if (buf[i] >= twoq) buf[i] -= twoq;
        if (buf[i] >= q)    buf[i] -= q;
    }
}

// Inverse NTT on one polynomial's CRT slot, matching ntt.rs non-AVX2.
static void host_ntt_inverse_1(uint64_t* buf, int crt_idx) {
    const uint32_t N    = SpiralParams::POLY_LEN;
    const int      logn = 11;
    const auto& inv  = g_host_ntt.inv[crt_idx];
    const auto& invp = g_host_ntt.invp[crt_idx];
    const uint64_t mod  = (crt_idx == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;
    const uint64_t twoq = 2 * mod;

    for (int mm = logn - 1; mm >= 0; --mm) {
        const uint32_t h = 1u << mm;
        const uint32_t t = N >> (mm + 1);
        for (uint32_t i = 0; i < h; ++i) {
            const uint64_t w  = inv[h + i];
            const uint64_t wp = invp[h + i];
            for (uint32_t j = 0; j < t; ++j) {
                const uint32_t upper = i * (2 * t) + j;
                const uint32_t lower = upper + t;
                uint64_t x = buf[upper];
                uint64_t y = buf[lower];
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

// NTT-transform a coefficient-domain polynomial (rows*cols*POLY_LEN raw u64 values
// mod LARGE_MOD) into NTT-domain buffer (rows*cols*CRT_COUNT*POLY_LEN u64 values).
// Output layout: modulus-major (CRT_COUNT*POLY_LEN per polynomial cell).
static void host_to_ntt(const uint64_t* raw,   // rows*cols*POLY_LEN source
                         uint64_t* ntt,          // rows*cols*CRT_COUNT*POLY_LEN dest
                         uint32_t rows, uint32_t cols) {
    const uint32_t N   = SpiralParams::POLY_LEN;
    const uint32_t CRT = SpiralParams::CRT_COUNT;
    const uint64_t cr1_0 = 68736257792ULL;  // Barrett cr1 for MODULUS_0
    const uint64_t cr1_1 = 73916747789ULL;  // Barrett cr1 for MODULUS_1

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint64_t* src = raw + (r * cols + c) * N;
            uint64_t* dst = ntt + (r * cols + c) * CRT * N;

            // CRT reduce: dst[0..N) = src[z] % q0, dst[N..2N) = src[z] % q1
            for (uint32_t z = 0; z < N; ++z) {
                dst[0 * N + z] = reduce_mod(src[z], SpiralParams::MODULUS_0, cr1_0);
                dst[1 * N + z] = reduce_mod(src[z], SpiralParams::MODULUS_1, cr1_1);
            }

            // NTT forward on each CRT slot
            host_ntt_forward_1(dst + 0 * N, 0);
            host_ntt_forward_1(dst + 1 * N, 1);
        }
    }
}

// ── Deserialization helpers ────────────────────────────────────────────────────

// Read one native-endian u64 from the byte stream.
static inline uint64_t read_u64_ne(const uint8_t* p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

// Deserialize a polynomial matrix, regenerating the first row from RNG.
// raw: pre-allocated buffer of rows*cols*POLY_LEN u64s (output).
// data: serialized bytes for (rows-1)*cols*POLY_LEN u64s.
// rng: ChaCha20 RNG positioned at the correct location.
static void deserialize_polymatrix_rng(uint64_t* raw,
                                        const uint8_t* data,
                                        uint32_t rows, uint32_t cols,
                                        ChaCha20Rng& rng)
{
    const uint32_t N  = SpiralParams::POLY_LEN;
    const uint64_t mod = LARGE_MOD;

    // First row: regenerated from RNG
    for (uint32_t i = 0; i < cols * N; ++i) {
        uint64_t r = rng.gen_u64();
        raw[i] = mod - (r % mod);  // get_inv_from_rng: mod - (rng % mod)
    }

    // Remaining rows: read from serialized bytes
    const uint32_t rest_words = (rows - 1) * cols * N;
    for (uint32_t i = 0; i < rest_words; ++i) {
        raw[cols * N + i] = read_u64_ne(data + i * 8);
    }
}

// Deserialize and upload a vector of polynomial matrices to the GPU.
// Each matrix has shape (rows × cols), and (rows-1)*cols polynomials are serialized.
static void deserialize_and_upload_vec(
    std::vector<DevicePolyMatrix>& out,
    const uint8_t*& cursor,
    uint32_t count, uint32_t rows, uint32_t cols,
    ChaCha20Rng& rng)
{
    const uint32_t N    = SpiralParams::POLY_LEN;
    const uint32_t CRT  = SpiralParams::CRT_COUNT;
    const size_t raw_words  = (size_t)rows * cols * N;
    const size_t ntt_words  = (size_t)rows * cols * CRT * N;
    const size_t serial_bytes = (size_t)(rows - 1) * cols * N * 8;

    build_host_ntt_tables();

    for (uint32_t k = 0; k < count; ++k) {
        // Allocate raw coefficient buffer on host
        std::vector<uint64_t> raw(raw_words);
        deserialize_polymatrix_rng(raw.data(), cursor, rows, cols, rng);
        cursor += serial_bytes;

        // CRT-reduce + NTT → NTT-domain buffer
        std::vector<uint64_t> ntt_buf(ntt_words);
        host_to_ntt(raw.data(), ntt_buf.data(), rows, cols);

        // Upload to device
        DevicePolyMatrix mat(rows, cols);
        mat.upload(ntt_buf.data(), ntt_words);
        out.push_back(std::move(mat));
    }
}

// ── parse_public_params ───────────────────────────────────────────────────────
//
// Mirrors PublicParameters::deserialize() from spiral-rs client.rs.
// Layout (expand_queries=true, version=0, t_exp_left != t_exp_right):
//   [32 bytes seed]
//   [v_packing:  N matrices of shape (N+1)×T_CONV, serialize rows-1=N rows]
//   [v_exp_left: g() matrices of shape 2×T_EXP_LEFT, serialize 1 row]
//   [v_exp_right: (stop_round()+1) matrices of shape 2×T_EXP_RIGHT, serialize 1 row]
//   [v_conversion: 1 matrix of shape 2×(2*T_CONV), serialize 1 row]

PublicParamsGPU parse_public_params(const uint8_t* data, size_t len,
                                    const SpiralParams& p) {
    if (len != p.setup_bytes()) {
        throw std::runtime_error("parse_public_params: length mismatch: got "
            + std::to_string(len) + ", expected " + std::to_string(p.setup_bytes()));
    }

    ensure_crt_constants();

    const uint8_t* cursor = data;

    // Read seed
    uint8_t seed[32];
    memcpy(seed, cursor, 32);
    cursor += 32;

    ChaCha20Rng rng;
    rng.init(seed);

    PublicParamsGPU pp;

    // v_packing: N matrices of shape (N+1) × T_CONV, serialize N rows
    deserialize_and_upload_vec(pp.v_packing, cursor,
        p.N, p.N + 1, p.T_CONV, rng);

    // v_expansion_left: g() matrices of shape 2 × T_EXP_LEFT, serialize 1 row
    deserialize_and_upload_vec(pp.v_expansion_left, cursor,
        p.g(), 2, p.T_EXP_LEFT, rng);

    // v_expansion_right: (stop_round()+1) matrices of shape 2 × T_EXP_RIGHT, 1 row
    deserialize_and_upload_vec(pp.v_expansion_right, cursor,
        p.stop_round() + 1, 2, p.T_EXP_RIGHT, rng);

    // v_conversion: 1 matrix of shape 2 × (2*T_CONV), serialize 1 row
    deserialize_and_upload_vec(pp.v_conversion, cursor,
        1, 2, 2 * p.T_CONV, rng);

    return pp;
}

// ── parse_query ───────────────────────────────────────────────────────────────
//
// Mirrors Query::deserialize() from spiral-rs client.rs (expand_queries=true).
// Layout:
//   [32 bytes seed]
//   [1 polynomial: 2×1 matrix, rows-1=1 row serialized = POLY_LEN u64s]

CiphertextGPU parse_query(const uint8_t* data, size_t len, const SpiralParams& p) {
    if (len != p.query_bytes()) {
        throw std::runtime_error("parse_query: length mismatch: got "
            + std::to_string(len) + ", expected " + std::to_string(p.query_bytes()));
    }

    ensure_crt_constants();
    build_host_ntt_tables();

    const uint8_t* cursor = data;
    uint8_t seed[32];
    memcpy(seed, cursor, 32);
    cursor += 32;

    ChaCha20Rng rng;
    rng.init(seed);

    const uint32_t N    = SpiralParams::POLY_LEN;
    const uint32_t CRT  = SpiralParams::CRT_COUNT;

    // 2×1 matrix, serialize rows-1=1 row (POLY_LEN u64s)
    std::vector<uint64_t> raw(2 * N);  // 2 rows × 1 col × POLY_LEN
    deserialize_polymatrix_rng(raw.data(), cursor, 2, 1, rng);

    // CRT-reduce + NTT
    std::vector<uint64_t> ntt_buf(2 * CRT * N);
    host_to_ntt(raw.data(), ntt_buf.data(), 2, 1);

    CiphertextGPU ct;
    ct.poly.upload(ntt_buf.data(), ntt_buf.size());
    return ct;
}

// ── encode_response ───────────────────────────────────────────────────────────
//
// Mirrors spiral-rs server.rs encode().
// result_mats[i] is the (N+1)×N packed ciphertext for instance i, in NTT domain.
//
// Steps:
//   1. INTT on device to get coefficient domain (CRT representation)
//   2. Download to host
//   3. CRT compose → single-modulus coefficients
//   4. Rescale first row to q2, rest rows to q1
//   5. Bit-pack into output bytes

std::vector<uint8_t> encode_response(const std::vector<DevicePolyMatrix>& result_mats,
                                     const SpiralParams& p) {
    ensure_crt_constants();
    build_host_ntt_tables();

    const uint32_t N       = SpiralParams::POLY_LEN;
    const uint32_t CRT     = SpiralParams::CRT_COUNT;
    const uint32_t instances = p.instances;
    const uint32_t rows    = p.N + 1;   // N+1
    const uint32_t cols    = p.N;       // N

    // q1 = 4 * P, q2 = Q2_VALUES[q2_bits]
    const uint64_t q1     = 4 * (uint64_t)p.P;
    const uint64_t q2     = SPIRAL_Q2_VALUES[p.Q2_BITS];
    const uint32_t q1_bits = (uint32_t)std::ceil(std::log2((double)q1));
    const uint32_t q2_bits = p.Q2_BITS;

    // Allocate result
    size_t num_bits = (size_t)instances
        * ((size_t)q2_bits * p.N * N
         + (size_t)q1_bits * p.N * p.N * N);
    constexpr size_t round_to = 64;
    size_t num_bytes = ((num_bits + round_to - 1) / round_to) * round_to / 8;
    std::vector<uint8_t> result(num_bytes, 0);

    size_t bit_offs = 0;
    const size_t ntt_words = (size_t)rows * cols * CRT * N;

    for (uint32_t inst = 0; inst < instances; ++inst) {
        const DevicePolyMatrix& mat = result_mats[inst];

        // 1. Copy to a mutable device buffer and run INTT
        uint64_t* d_copy = nullptr;
        cudaMalloc(&d_copy, ntt_words * sizeof(uint64_t));
        cudaMemcpy(d_copy, mat.d_data, ntt_words * sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice);
        launch_ntt_inverse(d_copy, rows * cols, 0 /*stream*/);
        cudaDeviceSynchronize();

        // 2. Download
        std::vector<uint64_t> host_ntt(ntt_words);
        cudaMemcpy(host_ntt.data(), d_copy, ntt_words * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_copy);

        // 3. CRT compose → raw[rows][cols][POLY_LEN]
        //    host_ntt layout: (row*cols + col) * CRT * N + crt*N + z
        std::vector<uint64_t> raw(rows * cols * N);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                const uint64_t* ptr = host_ntt.data() + (r * cols + c) * CRT * N;
                const uint64_t* crt0 = ptr + 0 * N;
                const uint64_t* crt1 = ptr + 1 * N;
                uint64_t* out = raw.data() + (r * cols + c) * N;
                for (uint32_t z = 0; z < N; ++z) {
                    out[z] = crt_compose(crt0[z], crt1[z]);
                }
            }
        }

        // 4+5. Rescale and bit-pack
        // first_row: submatrix row 0, columns 0..cols, POLY_LEN values each → q2
        // rest_rows: submatrix rows 1..(N+1), cols 0..N → q1
        // encode writes: N*POLY_LEN values at q2_bits, then N*N*POLY_LEN at q1_bits

        // First row: raw[0, c, z] for c in 0..N, z in 0..POLY_LEN
        for (uint32_t c = 0; c < cols; ++c) {
            const uint64_t* row0 = raw.data() + (0 * cols + c) * N;
            for (uint32_t z = 0; z < N; ++z) {
                uint64_t v = rescale_coeff(row0[z], LARGE_MOD, q2);
                write_arbitrary_bits(result.data(), v, bit_offs, q2_bits);
                bit_offs += q2_bits;
            }
        }

        // Rest rows: raw[r, c, z] for r in 1..rows, c in 0..cols, z in 0..POLY_LEN
        for (uint32_t r = 1; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                const uint64_t* row_r = raw.data() + (r * cols + c) * N;
                for (uint32_t z = 0; z < N; ++z) {
                    uint64_t v = rescale_coeff(row_r[z], LARGE_MOD, q1);
                    write_arbitrary_bits(result.data(), v, bit_offs, q1_bits);
                    bit_offs += q1_bits;
                }
            }
        }
    }

    return result;
}
