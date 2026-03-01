// NTT / INTT kernels for Spiral PIR.
//
// Mirrors spiral-rs src/ntt.rs ntt_forward / ntt_inverse (non-AVX2 path).
// Layout: modulus-major, data[crt * POLY_LEN + coeff] for a single polynomial.
//
// Table construction (host-side):
//   build_ntt_tables_host() computes:
//     [0] forward table   (root powers, bit-reversed)
//     [1] forward prime   (scaled: (inp << 32) / modulus, as u32 cast to u64)
//     [2] inverse table   (inv_root powers, bit-reversed, then div2_uint_mod applied)
//     [3] inverse prime   (scaled from inverse table)
//   and uploads them to device global memory via cudaMemcpy.
//
// Kernel strategy: one block per polynomial per CRT modulus.  With POLY_LEN=2048
// and 11 NTT stages, we use a single-pass iterative kernel that does all 11
// stages using shared memory (2048 u64s = 16 KB, within L1 limit).
//
// Reference:
//   res[0][2][0] == 134184961   (first element of MODULUS_0 inverse table)
//   res[0][2][1] == 96647580

#include "ntt.cuh"
#include "arith.cuh"
#include "params.hpp"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

// ── Device twiddle tables (global memory, not constant — size exceeds 64 KB) ──

// Four tables per modulus, allocated in init_ntt_tables().
// Indexed as table[m + i] matching the spiral-rs loop `forward_table[m + i]`.
__device__ uint64_t* d_ntt_fwd_0  = nullptr;   // forward root powers, mod MODULUS_0
__device__ uint64_t* d_ntt_fwd_1  = nullptr;   // forward root powers, mod MODULUS_1
__device__ uint64_t* d_ntt_fwdp_0 = nullptr;   // forward prime (scaled), mod MODULUS_0
__device__ uint64_t* d_ntt_fwdp_1 = nullptr;   // forward prime (scaled), mod MODULUS_1
__device__ uint64_t* d_ntt_inv_0  = nullptr;   // inverse root powers, mod MODULUS_0
__device__ uint64_t* d_ntt_inv_1  = nullptr;   // inverse root powers, mod MODULUS_1
__device__ uint64_t* d_ntt_invp_0 = nullptr;   // inverse prime (scaled), mod MODULUS_0
__device__ uint64_t* d_ntt_invp_1 = nullptr;   // inverse prime (scaled), mod MODULUS_1

// ── Host-side twiddle table construction ──────────────────────────────────────

// Modular exponentiation: base^exp mod modulus (u128 product for correctness).
static uint64_t mod_exp(uint64_t base, uint64_t exp, uint64_t modulus) {
    if (exp == 0) return 1;
    if (exp == 1) return base;
    uint64_t power  = base;
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(result) * power) % modulus);
        }
        exp >>= 1;
        if (exp > 0) {
            power = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(power) * power) % modulus);
        }
    }
    return result;
}

// Modular inverse via extended GCD (mirrors invert_uint_mod from number_theory.rs).
static uint64_t mod_inv(uint64_t value, uint64_t modulus) {
    // Extended Euclidean algorithm
    int64_t t = 0, newt = 1;
    int64_t r = static_cast<int64_t>(modulus), newr = static_cast<int64_t>(value);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp;
        tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (t < 0) t += static_cast<int64_t>(modulus);
    return static_cast<uint64_t>(t);
}

// Halve a value mod modulus (mirrors div2_uint_mod from arith.rs).
//   if odd: (val + modulus) >> 1   (handles potential overflow)
//   if even: val >> 1
static uint64_t div2_uint_mod(uint64_t val, uint64_t modulus) {
    if (val & 1) {
        // (val + modulus) may overflow u64 if both are large; use u128 to check
        unsigned __int128 sum = static_cast<unsigned __int128>(val) + modulus;
        uint64_t sum64 = static_cast<uint64_t>(sum);
        uint64_t carry = static_cast<uint64_t>(sum >> 64);
        return (sum64 >> 1) | (carry << 63);
    } else {
        return val >> 1;
    }
}

// Reverse the low `bit_count` bits of x (mirrors reverse_bits from arith.rs).
static uint64_t reverse_bits(uint64_t x, int bit_count) {
    if (bit_count == 0) return 0;
    // Reverse all 64 bits, then shift right.
    uint64_t r = 0;
    for (int i = 0; i < 64; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r >> (64 - bit_count);
}

// Find a primitive (2*poly_len)-th root of unity modulo `modulus`.
// Mirrors get_minimal_primitive_root(2 * poly_len, modulus) from number_theory.rs.
// These moduli are NTT-friendly: modulus ≡ 1 (mod 2*poly_len).
static uint64_t get_primitive_root_of_unity(uint64_t poly_len, uint64_t modulus) {
    // The 2*poly_len-th roots of unity form a subgroup of Z_modulus*.
    // Find a generator via: root = g^{(modulus-1) / (2*poly_len)}
    // where g is a generator of Z_modulus*.
    //
    // Use a deterministic search: try small candidate generators.
    uint64_t degree = 2 * poly_len;
    uint64_t size_quotient = (modulus - 1) / degree;

    // Iterate candidate primitive roots until we find one
    // where root^(degree/2) == modulus - 1.
    for (uint64_t g = 2; g < modulus; ++g) {
        uint64_t root = mod_exp(g, size_quotient, modulus);
        // Check is_primitive_root: root^(degree/2) == modulus - 1
        if (mod_exp(root, degree / 2, modulus) == modulus - 1) {
            // Now find the MINIMAL root (mirrors get_minimal_primitive_root):
            // Generate all powers root^(1), root^(3), root^(5), ... (odd powers)
            // — actually spiral-rs does: minimal = min over i=0..degree of
            //   current_generator, where we multiply by root^2 each step.
            uint64_t generator_sq = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(root) * root) % modulus);
            uint64_t current = root;
            uint64_t minimal = root;
            for (uint64_t i = 0; i < degree; ++i) {
                if (current < minimal) minimal = current;
                current = static_cast<uint64_t>(
                    (static_cast<unsigned __int128>(current) * generator_sq) % modulus);
            }
            return minimal;
        }
    }
    throw std::runtime_error("Failed to find primitive root of unity");
}

// Build one modulus's NTT tables (4 arrays of poly_len u64s).
// Returns [fwd, fwd_prime, inv, inv_prime].
// Mirrors build_ntt_tables from spiral-rs ntt.rs.
static std::vector<std::vector<uint64_t>> build_tables_for_modulus(
    uint64_t modulus, uint32_t poly_len)
{
    const int log_n = __builtin_ctz(poly_len);  // log2(poly_len) since power-of-2
    const uint32_t modulus_as_u32 = static_cast<uint32_t>(modulus);

    // Get primitive root and its inverse
    uint64_t root     = get_primitive_root_of_unity(poly_len, modulus);
    uint64_t inv_root = mod_inv(root, modulus);

    // --- Forward table: powers_of_primitive_root(root, modulus, log_n) ---
    // root_powers[reverse_bits(i, log_n)] = root^i  for i in 1..poly_len
    // root_powers[0] = 1
    std::vector<uint64_t> fwd(poly_len, 0);
    {
        uint64_t power = root;
        for (uint32_t i = 1; i < poly_len; ++i) {
            uint32_t idx = static_cast<uint32_t>(reverse_bits(i, log_n));
            fwd[idx] = power;
            power = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(power) * root) % modulus);
        }
        fwd[0] = 1;
    }

    // --- Forward prime: scale_powers_u32(modulus_as_u32, poly_len, fwd) ---
    // fwd_prime[i] = (uint32_t)((fwd[i] << 32) / modulus_as_u32)
    std::vector<uint64_t> fwd_prime(poly_len, 0);
    for (uint32_t i = 0; i < poly_len; ++i) {
        uint64_t wide_val = fwd[i] << 32;
        fwd_prime[i] = static_cast<uint32_t>(wide_val / modulus_as_u32);
    }

    // --- Inverse table: powers_of_primitive_root(inv_root, ...) with div2 ---
    // inv_root_powers[reverse_bits(i, log_n)] = inv_root^i  for i in 1..poly_len
    // inv_root_powers[0] = 1
    // Then apply div2_uint_mod to every element.
    std::vector<uint64_t> inv(poly_len, 0);
    {
        uint64_t power = inv_root;
        for (uint32_t i = 1; i < poly_len; ++i) {
            uint32_t idx = static_cast<uint32_t>(reverse_bits(i, log_n));
            inv[idx] = power;
            power = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(power) * inv_root) % modulus);
        }
        inv[0] = 1;
        for (uint32_t i = 0; i < poly_len; ++i) {
            inv[i] = div2_uint_mod(inv[i], modulus);
        }
    }

    // --- Inverse prime: scale_powers_u32(modulus_as_u32, poly_len, inv) ---
    std::vector<uint64_t> inv_prime(poly_len, 0);
    for (uint32_t i = 0; i < poly_len; ++i) {
        uint64_t wide_val = inv[i] << 32;
        inv_prime[i] = static_cast<uint32_t>(wide_val / modulus_as_u32);
    }

    return { fwd, fwd_prime, inv, inv_prime };
}

// ── init_ntt_tables: build host tables, allocate device memory, upload ────────

// Host-side copies (kept alive for potential re-init)
static uint64_t* h_ntt_fwd_0  = nullptr;
static uint64_t* h_ntt_fwdp_0 = nullptr;
static uint64_t* h_ntt_inv_0  = nullptr;
static uint64_t* h_ntt_invp_0 = nullptr;
static uint64_t* h_ntt_fwd_1  = nullptr;
static uint64_t* h_ntt_fwdp_1 = nullptr;
static uint64_t* h_ntt_inv_1  = nullptr;
static uint64_t* h_ntt_invp_1 = nullptr;

// Cached device-side pointers to the twiddle tables.
// Populated once in init_ntt_tables() via cudaMemcpyFromSymbol, then used
// directly in every launch_ntt_* call to avoid synchronizing symbol lookups.
static uint64_t* g_d_ntt_fwd_0  = nullptr;
static uint64_t* g_d_ntt_fwdp_0 = nullptr;
static uint64_t* g_d_ntt_inv_0  = nullptr;
static uint64_t* g_d_ntt_invp_0 = nullptr;
static uint64_t* g_d_ntt_fwd_1  = nullptr;
static uint64_t* g_d_ntt_fwdp_1 = nullptr;
static uint64_t* g_d_ntt_inv_1  = nullptr;
static uint64_t* g_d_ntt_invp_1 = nullptr;

// Helper to allocate + upload one twiddle array and set the device __device__ pointer.
static void upload_table(const std::vector<uint64_t>& host_table,
                          uint64_t** d_ptr_symbol,
                          uint64_t** h_keep)
{
    const size_t bytes = host_table.size() * sizeof(uint64_t);
    uint64_t* d_ptr = nullptr;
    cudaMalloc(&d_ptr, bytes);
    cudaMemcpy(d_ptr, host_table.data(), bytes, cudaMemcpyHostToDevice);
    // Write the device pointer into the __device__ global variable
    cudaMemcpyToSymbol(*d_ptr_symbol, &d_ptr, sizeof(uint64_t*));
    if (h_keep) {
        *h_keep = static_cast<uint64_t*>(malloc(bytes));
        memcpy(*h_keep, host_table.data(), bytes);
    }
}

void init_ntt_tables() {
    constexpr uint32_t N = SpiralParams::POLY_LEN;

    // Build tables for both moduli
    auto tables0 = build_tables_for_modulus(SpiralParams::MODULUS_0, N);
    auto tables1 = build_tables_for_modulus(SpiralParams::MODULUS_1, N);

    // Validate against known test vector from spiral-rs ntt.rs tests:
    //   res[0][2][0] == 134184961  (first element of MODULUS_0 inverse table)
    //   res[0][2][1] == 96647580
    if (tables0[2][0] != 134184961ULL || tables0[2][1] != 96647580ULL) {
        throw std::runtime_error(
            "NTT table validation failed: inverse table[0][0] or [0][1] mismatch");
    }

    // Allocate and upload to device
    upload_table(tables0[0], &d_ntt_fwd_0,  &h_ntt_fwd_0);
    upload_table(tables0[1], &d_ntt_fwdp_0, &h_ntt_fwdp_0);
    upload_table(tables0[2], &d_ntt_inv_0,  &h_ntt_inv_0);
    upload_table(tables0[3], &d_ntt_invp_0, &h_ntt_invp_0);
    upload_table(tables1[0], &d_ntt_fwd_1,  &h_ntt_fwd_1);
    upload_table(tables1[1], &d_ntt_fwdp_1, &h_ntt_fwdp_1);
    upload_table(tables1[2], &d_ntt_inv_1,  &h_ntt_inv_1);
    upload_table(tables1[3], &d_ntt_invp_1, &h_ntt_invp_1);

    // Cache device pointers on the host so launch_ntt_* never calls
    // cudaMemcpyFromSymbol in the hot path (it synchronizes the host).
    cudaMemcpyFromSymbol(&g_d_ntt_fwd_0,  d_ntt_fwd_0,  sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&g_d_ntt_fwdp_0, d_ntt_fwdp_0, sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&g_d_ntt_inv_0,  d_ntt_inv_0,  sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&g_d_ntt_invp_0, d_ntt_invp_0, sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&g_d_ntt_fwd_1,  d_ntt_fwd_1,  sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&g_d_ntt_fwdp_1, d_ntt_fwdp_1, sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&g_d_ntt_inv_1,  d_ntt_inv_1,  sizeof(uint64_t*));
    cudaMemcpyFromSymbol(&g_d_ntt_invp_1, d_ntt_invp_1, sizeof(uint64_t*));
}

// ── Forward NTT kernel ─────────────────────────────────────────────────────────
//
// One block per (polynomial, modulus) pair.
// blockDim.x = 1024 threads (handles 2048 coefficients, 2 per thread).
// Each block loads its polynomial into shared memory, performs 11 CT butterfly
// stages, then writes back.
//
// Spiral-rs forward butterfly (for one (i, j) pair in stage mm):
//   x = u32(op[j])
//   y = u32(op[j+t])
//   curr_x = x - 2q * (x >= 2q)   [lazy reduction]
//   q_tmp  = (y * w_prime) >> 32
//   q_new  = w * y - q_tmp * q
//   op[j]   = curr_x + q_new
//   op[j+t] = curr_x + 2q - q_new
// After all stages: two-step final reduction.

static constexpr uint32_t NTT_POLY_LEN  = SpiralParams::POLY_LEN;   // 2048
static constexpr uint32_t NTT_LOG_N     = 11;                        // log2(2048)
static constexpr uint32_t NTT_THREADS   = NTT_POLY_LEN / 2;         // 1024

__global__ void ntt_forward_kernel(
    uint64_t* __restrict__ d_polys,  // flat buffer: count * CRT_COUNT * POLY_LEN
    uint32_t count,
    const uint64_t* __restrict__ fwd0,
    const uint64_t* __restrict__ fwdp0,
    const uint64_t* __restrict__ fwd1,
    const uint64_t* __restrict__ fwdp1)
{
    // blockIdx.x = poly_index * CRT_COUNT + crt_index
    const uint32_t block_id = blockIdx.x;
    if (block_id >= count * SpiralParams::CRT_COUNT) return;

    const uint32_t poly_idx = block_id / SpiralParams::CRT_COUNT;
    const uint32_t crt_idx  = block_id % SpiralParams::CRT_COUNT;

    // Select tables and modulus for this CRT slot
    const uint64_t* fwd  = (crt_idx == 0) ? fwd0  : fwd1;
    const uint64_t* fwdp = (crt_idx == 0) ? fwdp0 : fwdp1;
    const uint64_t modulus = (crt_idx == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;
    const uint32_t modulus_u32      = static_cast<uint32_t>(modulus);
    const uint32_t two_times_mod_u32 = 2 * modulus_u32;

    // Base offset of this polynomial's CRT slot in the flat buffer
    const uint32_t base = (poly_idx * SpiralParams::CRT_COUNT + crt_idx) * NTT_POLY_LEN;

    // Load into shared memory
    __shared__ uint64_t s[NTT_POLY_LEN];
    const uint32_t tid = threadIdx.x;  // 0..1023
    s[tid]              = d_polys[base + tid];
    s[tid + NTT_THREADS] = d_polys[base + tid + NTT_THREADS];
    __syncthreads();

    // 11 CT butterfly stages: mm = 0..10
    for (uint32_t mm = 0; mm < NTT_LOG_N; ++mm) {
        const uint32_t m = 1u << mm;        // number of groups
        const uint32_t t = NTT_POLY_LEN >> (mm + 1);  // half-size of each group

        // Each thread handles one butterfly.
        // There are m*t = POLY_LEN/2 butterflies total.
        // Thread tid handles butterfly at:
        //   group i = tid / t
        //   position j = tid % t
        //   upper index = i * 2t + j
        //   lower index = i * 2t + j + t
        const uint32_t i = tid / t;
        const uint32_t j = tid % t;

        const uint64_t w       = fwd[m + i];
        const uint64_t w_prime = fwdp[m + i];

        const uint32_t upper = i * (2 * t) + j;
        const uint32_t lower = upper + t;

        const uint32_t x = static_cast<uint32_t>(s[upper]);
        const uint32_t y = static_cast<uint32_t>(s[lower]);

        const uint32_t curr_x = x - (two_times_mod_u32 * (x >= two_times_mod_u32));
        const uint64_t q_tmp  = (static_cast<uint64_t>(y) * w_prime) >> 32;
        // q_new = w * y - q_tmp * modulus   (w is the RAW root power; fwdp is scaled)
        const uint64_t q_new = w * static_cast<uint64_t>(y)
                               - q_tmp * static_cast<uint64_t>(modulus_u32);

        s[upper] = static_cast<uint64_t>(curr_x) + q_new;
        s[lower] = static_cast<uint64_t>(curr_x)
                   + static_cast<uint64_t>(two_times_mod_u32) - q_new;

        __syncthreads();
    }

    // Final two-step reduction: bring values from [0, 2q) to [0, q)
    const uint64_t twoq = static_cast<uint64_t>(two_times_mod_u32);
    const uint64_t q    = static_cast<uint64_t>(modulus_u32);

    uint64_t v0 = s[tid];
    v0 -= (v0 >= twoq) ? twoq : 0;
    v0 -= (v0 >= q)    ? q    : 0;
    s[tid] = v0;

    uint64_t v1 = s[tid + NTT_THREADS];
    v1 -= (v1 >= twoq) ? twoq : 0;
    v1 -= (v1 >= q)    ? q    : 0;
    s[tid + NTT_THREADS] = v1;

    __syncthreads();

    // Write back
    d_polys[base + tid]              = s[tid];
    d_polys[base + tid + NTT_THREADS] = s[tid + NTT_THREADS];
}

// ── Inverse NTT kernel ─────────────────────────────────────────────────────────
//
// Spiral-rs inverse butterfly (stage mm, iterated h=1<<mm from high to low):
//   x = op[j]
//   y = op[j+t]
//   t_tmp  = 2q - y + x
//   curr_x = x + y - 2q * ((x<<1) >= t_tmp)
//   h_tmp  = (t_tmp * w_prime) >> 32
//   res_x  = (curr_x + q * (t_tmp & 1)) >> 1     [div-by-2 at each stage]
//   res_y  = w * t_tmp - h_tmp * q
//   op[j]   = res_x
//   op[j+t] = res_y
// After all stages: two-step final reduction.
//
// Note: stages run in REVERSE order (mm = log_n-1 down to 0).

__global__ void ntt_inverse_kernel(
    uint64_t* __restrict__ d_polys,
    uint32_t count,
    const uint64_t* __restrict__ inv0,
    const uint64_t* __restrict__ invp0,
    const uint64_t* __restrict__ inv1,
    const uint64_t* __restrict__ invp1)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= count * SpiralParams::CRT_COUNT) return;

    const uint32_t poly_idx = block_id / SpiralParams::CRT_COUNT;
    const uint32_t crt_idx  = block_id % SpiralParams::CRT_COUNT;

    const uint64_t* inv  = (crt_idx == 0) ? inv0  : inv1;
    const uint64_t* invp = (crt_idx == 0) ? invp0 : invp1;
    const uint64_t modulus        = (crt_idx == 0) ? SpiralParams::MODULUS_0 : SpiralParams::MODULUS_1;
    const uint64_t two_times_mod  = 2 * modulus;

    const uint32_t base = (poly_idx * SpiralParams::CRT_COUNT + crt_idx) * NTT_POLY_LEN;

    __shared__ uint64_t s[NTT_POLY_LEN];
    const uint32_t tid = threadIdx.x;
    s[tid]               = d_polys[base + tid];
    s[tid + NTT_THREADS] = d_polys[base + tid + NTT_THREADS];
    __syncthreads();

    // Reverse stages: mm = log_n-1 downto 0
    for (int mm = NTT_LOG_N - 1; mm >= 0; --mm) {
        const uint32_t h = 1u << mm;
        const uint32_t t = NTT_POLY_LEN >> (mm + 1);

        // Thread tid handles butterfly i = tid/t, j = tid%t
        const uint32_t i = tid / t;
        const uint32_t j = tid % t;

        const uint64_t w      = inv[h + i];
        const uint64_t w_prime = invp[h + i];

        const uint32_t upper = i * (2 * t) + j;
        const uint32_t lower = upper + t;

        const uint64_t x = s[upper];
        const uint64_t y = s[lower];

        const uint64_t t_tmp  = two_times_mod - y + x;
        const uint64_t curr_x = x + y - (two_times_mod * (((x << 1) >= t_tmp) ? 1ULL : 0ULL));
        const uint64_t h_tmp  = (t_tmp * w_prime) >> 32;

        const uint64_t res_x  = (curr_x + modulus * (t_tmp & 1ULL)) >> 1;
        const uint64_t res_y  = w * t_tmp - h_tmp * modulus;

        s[upper] = res_x;
        s[lower] = res_y;

        __syncthreads();
    }

    // Final two-step reduction
    uint64_t v0 = s[tid];
    v0 -= (v0 >= two_times_mod) ? two_times_mod : 0;
    v0 -= (v0 >= modulus)       ? modulus        : 0;
    s[tid] = v0;

    uint64_t v1 = s[tid + NTT_THREADS];
    v1 -= (v1 >= two_times_mod) ? two_times_mod : 0;
    v1 -= (v1 >= modulus)       ? modulus        : 0;
    s[tid + NTT_THREADS] = v1;

    __syncthreads();

    d_polys[base + tid]               = s[tid];
    d_polys[base + tid + NTT_THREADS] = s[tid + NTT_THREADS];
}

// ── Public launch wrappers ─────────────────────────────────────────────────────

void launch_ntt_forward(uint64_t* d_polys, uint32_t count, cudaStream_t stream) {
    if (count == 0) return;
    const uint32_t num_blocks = count * SpiralParams::CRT_COUNT;
    ntt_forward_kernel<<<num_blocks, NTT_THREADS, 0, stream>>>(
        d_polys, count,
        g_d_ntt_fwd_0, g_d_ntt_fwdp_0, g_d_ntt_fwd_1, g_d_ntt_fwdp_1);
}

void launch_ntt_inverse(uint64_t* d_polys, uint32_t count, cudaStream_t stream) {
    if (count == 0) return;
    const uint32_t num_blocks = count * SpiralParams::CRT_COUNT;
    ntt_inverse_kernel<<<num_blocks, NTT_THREADS, 0, stream>>>(
        d_polys, count,
        g_d_ntt_inv_0, g_d_ntt_invp_0, g_d_ntt_inv_1, g_d_ntt_invp_1);
}
