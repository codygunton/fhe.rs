// Query coefficient expansion for Spiral PIR.
//
// Implements coefficient_expansion(), regev_to_gsw(), and reorient_reg_ciphertexts()
// on GPU, mirroring spiral-rs server.rs expand_query().
//
// Pipeline:
//   1. coefficient_expansion: expand 1 query ciphertext → (1<<g) NTT ciphertexts
//   2. Split results into v_reg_inp (dim0 ciphertexts) and v_gsw_inp (T_GSW*nu_2)
//   3. reorient_reg_ciphertexts: pack into CRT-packed reoriented flat buffer
//   4. regev_to_gsw: convert T_GSW regev ciphertexts per GSW key into v_folding

#include "types.hpp"
#include "params.hpp"
#include "kernels/ntt.cuh"
#include "kernels/automorph.cuh"
#include "kernels/gadget.cuh"
#include "kernels/poly_ops.cuh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

// ── Helper: bits_per ─────────────────────────────────────────────────────────
// Mirrors spiral-rs gadget.rs get_bits_per(params, dim):
//   modulus_log2 = 56,  dim = num_elems
//   if dim == modulus_log2: return 1
//   else: floor(56.0 / dim) + 1
static uint32_t get_bits_per(uint32_t dim) {
    constexpr uint32_t MOD_LOG2 = 56;
    if (dim == MOD_LOG2) return 1;
    return static_cast<uint32_t>(static_cast<double>(MOD_LOG2) / dim) + 1;
}

// ── Helper: allocate and zero a DevicePolyMatrix ──────────────────────────────
static DevicePolyMatrix alloc_zero(uint32_t rows, uint32_t cols, cudaStream_t stream) {
    DevicePolyMatrix m(rows, cols);
    cudaMemsetAsync(m.d_data, 0, m.byte_size(), stream);
    return m;
}

// ── Helper: copy columns from src into dst starting at dst column dst_col ─────
// Mirrors spiral-rs PolyMatrixNTT::copy_into(src, row_offset, col_offset).
// Copies all rows of src into dst at column offset dst_col_off.
// ── Helper: scalar multiply d_out = neg1[r] * d_in ────────────────────────────
// neg1[r] is a 1×1 NTT polynomial matrix = NTT(-X^{N/2^{r+1}}).
// Mirrors spiral-rs get_v_neg1: v_neg1[r] has coefficient N/2^{r+1} = -1.
// Builds neg1 on GPU: coefficient N/2^{r+1} has value (qi - 1) for each crt slot,
// all others 0, then NTT forward.
// Returns a 1×1 DevicePolyMatrix in NTT domain.
static DevicePolyMatrix build_v_neg1_entry(uint32_t r, cudaStream_t stream) {
    const uint32_t N   = SpiralParams::POLY_LEN;
    const uint32_t CRT = SpiralParams::CRT_COUNT;
    const uint64_t mods[2] = { SpiralParams::MODULUS_0, SpiralParams::MODULUS_1 };

    DevicePolyMatrix neg1 = alloc_zero(1, 1, stream);

    // Position: N - 2^r, matching spiral-rs get_v_neg1
    // where idx = poly_len - (1 << i) for round i=r.
    const uint32_t pos = N - (1u << r);

    // The neg1 polynomial is -X^pos, representing coefficient -1 at position pos.
    // In Rust get_v_neg1: raw coefficient = LARGE_MOD - 1 (= -1 mod q0*q1).
    // CRT-reduced: CRT slot k = (LARGE_MOD - 1) % q_k = q_k - 1.
    // Both CRT slots get their respective (modulus - 1), NOT (q0-1) % q1.
    std::vector<uint64_t> host_buf(CRT * N, 0ULL);
    for (uint32_t crt = 0; crt < CRT; ++crt) {
        host_buf[crt * N + pos] = mods[crt] - 1ULL;
    }
    cudaMemcpyAsync(neg1.d_data, host_buf.data(), CRT * N * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream);

    // NTT forward (count = 1 polynomial matrix = 1*1 = 1 poly per crt slot, total CRT)
    launch_ntt_forward(neg1.d_data, 1, stream);

    return neg1;
}

// ── Reorient kernel ───────────────────────────────────────────────────────────
// Mirrors reorient_reg_ciphertexts from spiral-rs util.rs.
// v_reg[j] is a 2×1 NTT matrix.
// Output: out[z * dim0*2 + j*2 + r] = (v_reg[j][crt0,z] & 0xFFFF) | (v_reg[j][crt1,z] << 32)
// We do this on the host after downloading v_reg[j] from GPU.
// (dim0 is typically 512, N=2048 → 512*2*2048 = 2M u64s → 16 MB, fine to do on host)

static std::vector<uint64_t> reorient_reg_gpu(
    const std::vector<DevicePolyMatrix>& v_reg,
    uint32_t dim0,
    cudaStream_t stream)
{
    const uint32_t N   = SpiralParams::POLY_LEN;
    const uint32_t CRT = SpiralParams::CRT_COUNT;
    const size_t v_reg_sz = static_cast<size_t>(dim0) * 2 * N;
    std::vector<uint64_t> out(v_reg_sz, 0ULL);

    // Download each v_reg[j] and pack into out
    std::vector<uint64_t> tmp(2 * CRT * N);
    for (uint32_t j = 0; j < dim0; ++j) {
        // Sync copy (reorient is called once, at the end of expansion)
        cudaMemcpyAsync(tmp.data(), v_reg[j].d_data, 2 * CRT * N * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // v_reg[j] modulus-major layout:
        //   data[row * CRT * N + crt * N + z]
        // row=0 → r=0, row=1 → r=1
        for (uint32_t r = 0; r < 2; ++r) {
            for (uint32_t z = 0; z < N; ++z) {
                uint64_t val0 = tmp[r * CRT * N + 0 * N + z];  // crt0
                uint64_t val1 = tmp[r * CRT * N + 1 * N + z];  // crt1
                val0 %= SpiralParams::MODULUS_0;
                val1 %= SpiralParams::MODULUS_1;
                const size_t idx_out = static_cast<size_t>(z) * dim0 * 2 + j * 2 + r;
                out[idx_out] = (val0 & 0xFFFFFFFFULL) | (val1 << 32);
            }
        }
    }
    return out;
}

// ── coefficient_expansion ─────────────────────────────────────────────────────
// GPU port of spiral-rs server.rs coefficient_expansion().
//
// On entry:  v[0] = query ciphertext (2×1 NTT), v[1..(1<<g)-1] = zero
// On exit:   v[0..g] expanded such that only v[target] encodes scale_k,
//            v[1,3,5,...] encode the GSW dimension coefficients.
//
// Algorithm per round r ∈ [0, g):
//   1. For i ∈ [0, num_in): d = scalar_multiply(neg1[r], s[i])  → dest[i]
//   2. For i ∈ [0, num_in) (and num_in..num_out):
//      a. INTT(v[i]) → ct (coefficient domain)
//      b. automorph(ct, t) → ct_auto
//      c. gadget_invert_crt(ct_auto row 0, rdim=1) → gi_ct
//      d. NTT(gi_ct) → gi_ct_ntt
//      e. ct_auto_1 = ct_auto.row(1) → NTT
//      f. w_times_ginv = w[r] × gi_ct_ntt  (2×T_EXP × T_EXP×1 = 2×1)
//      g. v[i] += w_times_ginv + ct_auto_1 (broadcast to row 0 and row 1)

static void coefficient_expansion_gpu(
    std::vector<DevicePolyMatrix>& v,
    uint32_t g_rounds,
    uint32_t stop_round,
    const PublicParamsGPU& pp,
    const std::vector<DevicePolyMatrix>& v_neg1,
    uint32_t max_bits_right,
    cudaStream_t stream)
{
    const uint32_t N   = SpiralParams::POLY_LEN;
    const uint32_t CRT = SpiralParams::CRT_COUNT;

    for (uint32_t r = 0; r < g_rounds; ++r) {
        const uint32_t num_in  = 1u << r;
        const uint32_t num_out = 2 * num_in;
        const uint32_t t       = (N / num_in) + 1;  // automorphism element

        // Choose which expansion key to use for left vs right
        // right: r=0 always, or r>0 && i%2==1
        // left:  r>0 && i%2==0

        // Step 1: scalar_multiply dest[i] = neg1[r] * src[i] for i in [0, num_in)
        // neg1[r] is a 1×1 matrix, src[i] is a 2×1 matrix
        // We need to multiply pointwise (broadcast over rows):
        //   dest[i][row, col, crt, z] = neg1[r][0, 0, crt, z] * src[i][row, col, crt, z]
        for (uint32_t i = 0; i < num_in; ++i) {
            launch_poly_scalar_mul(v[num_in + i].d_data, v[i].d_data,
                                   v_neg1[r].d_data, 2, 1, stream);
        }

        // Step 2: expand each entry
        for (uint32_t i = 0; i < num_out; ++i) {
            // Skip entries that don't need expansion (right side, beyond stop_round)
            if (stop_round > 0 && r > stop_round && (i % 2) == 1) continue;
            if (stop_round > 0 && r == stop_round && (i % 2) == 1
                && (i / 2) >= max_bits_right) continue;

            // Select key
            const DevicePolyMatrix& w = ((r != 0) && (i % 2 == 0))
                ? pp.v_expansion_left[r]
                : pp.v_expansion_right[r];

            // bits_per / num_elems for this key
            uint32_t t_exp = ((r != 0) && (i % 2 == 0))
                ? SpiralParams::T_EXP_LEFT
                : SpiralParams::T_EXP_RIGHT;
            uint32_t num_elems = t_exp;
            uint32_t bits_per_val = get_bits_per(num_elems);

            // a. INTT v[i] → ct (coefficient domain); count=rows*cols=2 for a 2×1 matrix
            DevicePolyMatrix ct = alloc_zero(2, 1, stream);
            cudaMemcpyAsync(ct.d_data, v[i].d_data, v[i].byte_size(),
                            cudaMemcpyDeviceToDevice, stream);
            launch_ntt_inverse(ct.d_data, 2, stream);

            // b. automorph(ct, t) → ct_auto; output buffer must be zeroed
            DevicePolyMatrix ct_auto = alloc_zero(2, 1, stream);
            launch_automorph(ct_auto.d_data, ct.d_data, t, 2, stream);

            // c. gadget_invert_crt on row 0 of ct_auto (rdim=1, cols=1)
            //    Input: 1×1 slice of ct_auto (row 0)
            //    Output: T_EXP × 1 coefficient-domain matrix
            DevicePolyMatrix gi_ct = alloc_zero(num_elems, 1, stream);
            // Row 0 of ct_auto is at ct_auto.d_data[0 * CRT * N ...]
            launch_gadget_invert_crt(gi_ct.d_data, ct_auto.d_data,
                                     1, 1, num_elems, bits_per_val, stream);

            // d. NTT(gi_ct) → gi_ct_ntt  (to_ntt_no_reduce: just forward NTT)
            launch_ntt_forward(gi_ct.d_data, num_elems, stream);

            // e. ct_auto_1 = ct_auto row 1 → NTT domain 1×1 matrix
            DevicePolyMatrix ct_auto_1 = alloc_zero(1, 1, stream);
            // Row 1 of ct_auto: offset = 1 * CRT * N
            cudaMemcpyAsync(ct_auto_1.d_data,
                            ct_auto.d_data + 1 * CRT * N,
                            CRT * N * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);
            // ct_auto_1 is already in coefficient domain — NTT it
            launch_ntt_forward(ct_auto_1.d_data, 1, stream);

            // f. w_times_ginv = w × gi_ct  → 2×1
            DevicePolyMatrix w_ginv = alloc_zero(2, 1, stream);
            launch_poly_mat_mul(w_ginv.d_data, w.d_data, 2, num_elems,
                                gi_ct.d_data, 1, stream);

            // g. v[i] += w_ginv; then add ct_auto_1 only to row 1
            // Mirrors spiral-rs: sum += j * ct_auto_1 where j=0 for row0, j=1 for row1
            launch_poly_add_inplace(v[i].d_data, w_ginv.d_data, 2, 1, stream);
            // Add ct_auto_1 only to row 1 of v[i]
            launch_poly_add_inplace(v[i].d_data + 1 * CRT * N,
                                    ct_auto_1.d_data, 1, 1, stream);
        }
    }
}

// ── regev_to_gsw ─────────────────────────────────────────────────────────────
// GPU port of spiral-rs server.rs regev_to_gsw().
// v_conversion: 2 × (2*T_CONV) NTT matrix
// v_inp[i]:     2×1 NTT matrix (T_GSW per output GSW)
// Output: v_gsw[k] = 2 × (2*T_GSW) NTT matrix
static void regev_to_gsw_gpu(
    std::vector<DevicePolyMatrix>& v_gsw,
    const std::vector<DevicePolyMatrix>& v_inp,
    const DevicePolyMatrix& v_conv,
    const SpiralParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t N      = SpiralParams::POLY_LEN;
    constexpr uint32_t CRT    = SpiralParams::CRT_COUNT;
    const uint32_t T_GSW  = SpiralParams::T_GSW;
    const uint32_t T_CONV = SpiralParams::T_CONV;

    // bits_per for gadget_invert with 2*T_CONV limbs
    const uint32_t num_elems_conv = 2 * T_CONV;
    const uint32_t bits_per_conv  = get_bits_per(num_elems_conv);

    for (size_t i = 0; i < v_gsw.size(); ++i) {
        DevicePolyMatrix& ct = v_gsw[i];
        // ct is 2 × (2*T_GSW), already zeroed

        for (uint32_t j = 0; j < T_GSW; ++j) {
            const uint32_t idx_inp = i * T_GSW + j;
            const DevicePolyMatrix& inp = v_inp[idx_inp];

            // Copy inp into ct odd column: ct.col(2*j+1) = inp.col(0)
            // ct is 2 × (2*T_GSW), each "column" = all rows for that col index
            // For a 2×(2*T_GSW) matrix, poly at (row, col) offset = (row*(2*T_GSW)+col)*CRT*N
            for (uint32_t row = 0; row < 2; ++row) {
                const size_t src_off = static_cast<size_t>(row * 1 + 0) * CRT * N;
                const size_t dst_off = static_cast<size_t>(row * (2*T_GSW) + (2*j+1)) * CRT * N;
                cudaMemcpyAsync(ct.d_data + dst_off, inp.d_data + src_off,
                                CRT * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            }

            // INTT(inp) → tmp_ct_raw; count=rows*cols=2 to convert both rows
            DevicePolyMatrix tmp_raw(2, 1);
            cudaMemcpyAsync(tmp_raw.d_data, inp.d_data, inp.byte_size(),
                            cudaMemcpyDeviceToDevice, stream);
            launch_ntt_inverse(tmp_raw.d_data, 2, stream);

            // gadget_invert(tmp_raw, rdim=2) → ginv_c_inp  (2*T_CONV × 1, coeff domain)
            DevicePolyMatrix ginv(num_elems_conv * 2, 1);
            cudaMemsetAsync(ginv.d_data, 0, ginv.byte_size(), stream);
            // rdim=2 (full rows), num_elems = T_CONV (per row), total out_rows = 2*T_CONV
            // BUT gadget_invert with rdim=inp.rows means:
            //   num_elems = out.rows / rdim = 2*T_CONV / 2 = T_CONV
            // bits_per = get_bits_per(T_CONV)
            const uint32_t bits_per_ginv = get_bits_per(T_CONV);
            launch_gadget_invert_crt(ginv.d_data, tmp_raw.d_data,
                                     2, 1, T_CONV, bits_per_ginv, stream);

            // NTT(ginv) → ginv_ntt
            launch_ntt_forward(ginv.d_data, num_elems_conv, stream);

            // multiply: tmp_ct = v_conv × ginv_ntt  (2 × 2*T_CONV) × (2*T_CONV × 1) = 2×1
            DevicePolyMatrix tmp_ct(2, 1);
            cudaMemsetAsync(tmp_ct.d_data, 0, tmp_ct.byte_size(), stream);
            launch_poly_mat_mul(tmp_ct.d_data, v_conv.d_data, 2, num_elems_conv,
                                ginv.d_data, 1, stream);

            // Copy tmp_ct into ct even column: ct.col(2*j) = tmp_ct.col(0)
            for (uint32_t row = 0; row < 2; ++row) {
                const size_t src_off = static_cast<size_t>(row * 1 + 0) * CRT * N;
                const size_t dst_off = static_cast<size_t>(row * (2*T_GSW) + (2*j)) * CRT * N;
                cudaMemcpyAsync(ct.d_data + dst_off, tmp_ct.d_data + src_off,
                                CRT * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            }
        }
    }
}

// ── expand_query_gpu ──────────────────────────────────────────────────────────

std::pair<std::vector<uint64_t>, std::vector<DevicePolyMatrix>>
expand_query_gpu(const CiphertextGPU& ct, const PublicParamsGPU& pp,
                 const SpiralParams& p, cudaStream_t stream)
{
    const uint32_t dim0          = p.dim0();
    const uint32_t nu_2          = p.nu_2;
    const uint32_t T_GSW         = SpiralParams::T_GSW;
    const uint32_t g_rounds      = p.g();
    const uint32_t stop_round    = p.stop_round();
    const uint32_t right_expanded = T_GSW * nu_2;
    const uint32_t max_bits_right = T_GSW * nu_2;

    // Allocate expansion tree: 2^g entries of 2×1 NTT matrices
    const uint32_t tree_size = 1u << g_rounds;
    std::vector<DevicePolyMatrix> v;
    v.reserve(tree_size);
    for (uint32_t i = 0; i < tree_size; ++i) {
        v.push_back(alloc_zero(2, 1, stream));
    }

    // Copy query ciphertext into v[0]
    cudaMemcpyAsync(v[0].d_data, ct.poly.d_data, ct.poly.byte_size(),
                    cudaMemcpyDeviceToDevice, stream);

    // Build v_neg1 entries (one per round)
    std::vector<DevicePolyMatrix> v_neg1;
    v_neg1.reserve(g_rounds);
    for (uint32_t r = 0; r < g_rounds; ++r) {
        v_neg1.push_back(build_v_neg1_entry(r, stream));
    }

    // Run coefficient expansion
    coefficient_expansion_gpu(v, g_rounds, stop_round, pp, v_neg1, max_bits_right, stream);

    // Split: v_reg_inp = v[0, 2, 4, ... 2*(dim0-1)]
    //        v_gsw_inp = v[1, 3, 5, ... 2*(right_expanded-1)+1]
    std::vector<DevicePolyMatrix> v_reg_inp;
    v_reg_inp.reserve(dim0);
    for (uint32_t i = 0; i < dim0; ++i) {
        DevicePolyMatrix m(2, 1);
        cudaMemcpyAsync(m.d_data, v[2 * i].d_data, v[2*i].byte_size(),
                        cudaMemcpyDeviceToDevice, stream);
        v_reg_inp.push_back(std::move(m));
    }

    std::vector<DevicePolyMatrix> v_gsw_inp;
    v_gsw_inp.reserve(right_expanded);
    for (uint32_t i = 0; i < right_expanded; ++i) {
        DevicePolyMatrix m(2, 1);
        cudaMemcpyAsync(m.d_data, v[2 * i + 1].d_data, v[2*i+1].byte_size(),
                        cudaMemcpyDeviceToDevice, stream);
        v_gsw_inp.push_back(std::move(m));
    }

    // Reorient reg ciphertexts → flat CRT-packed buffer on host
    std::vector<uint64_t> v_reg_reoriented = reorient_reg_gpu(v_reg_inp, dim0, stream);

    // Build v_folding via regev_to_gsw
    std::vector<DevicePolyMatrix> v_folding;
    v_folding.reserve(nu_2);
    for (uint32_t k = 0; k < nu_2; ++k) {
        v_folding.push_back(alloc_zero(2, 2 * T_GSW, stream));
    }

    const DevicePolyMatrix& v_conv = pp.v_conversion[0];

    regev_to_gsw_gpu(v_folding, v_gsw_inp, v_conv, p, stream);

    cudaStreamSynchronize(stream);

    return { std::move(v_reg_reoriented), std::move(v_folding) };
}
