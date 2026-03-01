// Second-dimension fold and packing for Spiral PIR.
//
// Mirrors spiral-rs server.rs fold_ciphertexts(), get_v_folding_neg(), and pack().
//
// fold_ciphertexts: reduces num_per ciphertexts → 1 using nu_2 GSW folding keys.
//   Each round halves the count using GSW-based CMux:
//     out[i] = v_folding_neg * ginv(cts[i]) + v_folding * ginv(cts[num_per+i])
//
// pack: accumulates N×N ciphertexts into one (N+1)×N result:
//   for each col c, for each row r:
//     v_int += w[r] * gadget_invert(ct[r*N+c].row0)
//     v_int.row(1+r) += ct[r*N+c].row1 (in NTT domain)

#include "types.hpp"
#include "params.hpp"
#include "kernels/ntt.cuh"
#include "kernels/gadget.cuh"
#include "kernels/poly_ops.cuh"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

// ── Helper: bits_per ──────────────────────────────────────────────────────────
static uint32_t get_bits_per(uint32_t dim) {
    constexpr uint32_t MOD_LOG2 = 56;
    if (dim == MOD_LOG2) return 1;
    return static_cast<uint32_t>(static_cast<double>(MOD_LOG2) / dim) + 1;
}

// ── Helper: allocate and zero ─────────────────────────────────────────────────
static DevicePolyMatrix alloc_zero(uint32_t rows, uint32_t cols, cudaStream_t stream) {
    DevicePolyMatrix m(rows, cols);
    cudaMemsetAsync(m.d_data, 0, m.byte_size(), stream);
    return m;
}

// ── build_gadget_ntt ──────────────────────────────────────────────────────────
// Builds the NTT of gadget(2, 2*T_GSW) on GPU.
// build_gadget(rows=2, cols=2*T_GSW):
//   For i in 0..rows, j in 0..num_elems (=T_GSW):
//     poly[i, i+j*rows][0] = 1 << (bits_per * j)
// Since T_GSW=7, bits_per = floor(56/7)+1 = 9 (if 7 != 56), coeff[0]=2^{9j}
// The gadget polynomials are single-coefficient constants.
// NTT of a constant c is (c, c, c, ...) in NTT domain (since X^0=1 maps to all 1).
// Actually NTT of f(X) where f has only constant term c:
//   NTT[z] = f(omega^z) = c * 1 = c
// So gadget_ntt is a constant-filled matrix.
static DevicePolyMatrix build_gadget_ntt(uint32_t rows, uint32_t cols, cudaStream_t stream) {
    const uint32_t N   = SpiralParams::POLY_LEN;
    const uint32_t CRT = SpiralParams::CRT_COUNT;

    DevicePolyMatrix g = alloc_zero(rows, cols, stream);

    assert(cols % rows == 0);
    const uint32_t num_elems = cols / rows;
    const uint32_t bits_per  = get_bits_per(num_elems);
    const uint64_t mods[2] = { SpiralParams::MODULUS_0, SpiralParams::MODULUS_1 };

    std::vector<uint64_t> host_buf(rows * cols * CRT * N, 0ULL);

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < num_elems; ++j) {
            if (bits_per * j >= 64) continue;
            const uint64_t val = 1ULL << (bits_per * j);
            const uint32_t poly_col = i + j * rows;  // column index in gadget matrix
            // Poly at (row=i, col=poly_col) offset = (i*cols + poly_col) * CRT * N
            const size_t poly_off = static_cast<size_t>(i * cols + poly_col) * CRT * N;
            for (uint32_t crt = 0; crt < CRT; ++crt) {
                // Constant-term only polynomial: NTT = all coefficients equal to val mod qi
                // (NTT of constant polynomial c is [c, c, c, ...])
                uint64_t v_mod = val % mods[crt];
                for (uint32_t z = 0; z < N; ++z) {
                    host_buf[poly_off + crt * N + z] = v_mod;
                }
            }
        }
    }

    cudaMemcpyAsync(g.d_data, host_buf.data(), host_buf.size() * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream);
    // Already in NTT domain (all-constant fill = NTT of constant polynomial)
    return g;
}

// ── compute_v_folding_neg ─────────────────────────────────────────────────────
// v_folding_neg[i] = gadget_ntt - v_folding[i]
//                  = gadget_ntt + NTT(-INTT(v_folding[i]))
//
// Mirrors spiral-rs get_v_folding_neg():
//   invert(ct_gsw_inv, v_folding[i].raw())  → ct_gsw_inv = -v_folding[i] (coeff domain)
//   ct_gsw_neg = gadget_ntt + ntt(ct_gsw_inv)
//
// "invert" in spiral-rs: negate all coefficients mod q.
// We implement this as: INTT(v_folding[i]) → negate → NTT → add gadget_ntt
static std::vector<DevicePolyMatrix> compute_v_folding_neg(
    const std::vector<DevicePolyMatrix>& v_folding,
    const DevicePolyMatrix& gadget_ntt,
    cudaStream_t stream)
{
    std::vector<DevicePolyMatrix> v_neg;
    v_neg.reserve(v_folding.size());

    for (const DevicePolyMatrix& vf : v_folding) {
        const uint32_t rows = vf.rows;
        const uint32_t cols = vf.cols;

        // Copy and INTT into coefficient domain
        DevicePolyMatrix tmp(rows, cols);
        cudaMemcpyAsync(tmp.d_data, vf.d_data, vf.byte_size(),
                        cudaMemcpyDeviceToDevice, stream);
        launch_ntt_inverse(tmp.d_data, rows * cols, stream);

        // Negate all coefficients on GPU (no CPU round-trip, no stream sync)
        launch_poly_negate(tmp.d_data, rows, cols, stream);

        // NTT forward → negated key in NTT domain
        launch_ntt_forward(tmp.d_data, rows * cols, stream);

        // Add gadget_ntt: ct_gsw_neg = gadget_ntt + ntt(ct_gsw_inv)
        DevicePolyMatrix neg(rows, cols);
        cudaMemcpyAsync(neg.d_data, gadget_ntt.d_data, gadget_ntt.byte_size(),
                        cudaMemcpyDeviceToDevice, stream);
        launch_poly_add_inplace(neg.d_data, tmp.d_data, rows, cols, stream);

        v_neg.push_back(std::move(neg));
    }

    return v_neg;
}

// ── fold_ciphertexts_gpu ──────────────────────────────────────────────────────

void fold_ciphertexts_gpu(std::vector<DevicePolyMatrix>& v_cts,
                          const std::vector<DevicePolyMatrix>& v_gsw,
                          const SpiralParams& p, cudaStream_t stream)
{
    if (v_cts.size() == 1) return;

    const uint32_t T_GSW  = SpiralParams::T_GSW;
    const uint32_t nu_2   = p.nu_2;
    const uint32_t further_dims = nu_2;  // log2(num_per)

    // ell = v_gsw[0].cols / 2 = (2*T_GSW) / 2 = T_GSW
    const uint32_t ell = v_gsw[0].cols / 2;

    // Build gadget_ntt for compute_v_folding_neg
    DevicePolyMatrix gadget_ntt = build_gadget_ntt(2, 2 * T_GSW, stream);

    // Compute v_folding_neg
    std::vector<DevicePolyMatrix> v_folding_neg =
        compute_v_folding_neg(v_gsw, gadget_ntt, stream);

    // Gadget params for ginv of a 2×1 ciphertext (rdim = 2):
    //   num_elems = ell = T_GSW, bits_per = get_bits_per(T_GSW)
    const uint32_t bits_per = get_bits_per(ell);

    uint32_t cur_num = static_cast<uint32_t>(v_cts.size());

    for (uint32_t cur_dim = 0; cur_dim < further_dims; ++cur_dim) {
        cur_num = cur_num / 2;
        const uint32_t key_idx = further_dims - 1 - cur_dim;

        for (uint32_t i = 0; i < cur_num; ++i) {
            // v_cts[i] is already in coefficient domain (INTT done before fold)
            DevicePolyMatrix ct_i(2, 1);
            cudaMemcpyAsync(ct_i.d_data, v_cts[i].d_data, v_cts[i].byte_size(),
                            cudaMemcpyDeviceToDevice, stream);

            // gadget_invert(ct_i, rdim=2) → ginv_c  (2*ell × 1)
            DevicePolyMatrix ginv_c = alloc_zero(2 * ell, 1, stream);
            launch_gadget_invert_crt(ginv_c.d_data, ct_i.d_data,
                                     2, 1, ell, bits_per, stream);

            // NTT(ginv_c)
            launch_ntt_forward(ginv_c.d_data, 2 * ell, stream);

            // prod = v_folding_neg[key_idx] × ginv_c  (2×2*T_GSW) × (2*T_GSW×1) = 2×1
            DevicePolyMatrix prod = alloc_zero(2, 1, stream);
            launch_poly_mat_mul(prod.d_data,
                                v_folding_neg[key_idx].d_data, 2, 2 * ell,
                                ginv_c.d_data, 1, stream);

            // v_cts[cur_num+i] is already in coefficient domain
            DevicePolyMatrix ct_j(2, 1);
            cudaMemcpyAsync(ct_j.d_data, v_cts[cur_num + i].d_data, v_cts[cur_num+i].byte_size(),
                            cudaMemcpyDeviceToDevice, stream);

            // gadget_invert(ct_j, rdim=2) → ginv_c
            DevicePolyMatrix ginv_c2 = alloc_zero(2 * ell, 1, stream);
            launch_gadget_invert_crt(ginv_c2.d_data, ct_j.d_data,
                                     2, 1, ell, bits_per, stream);
            launch_ntt_forward(ginv_c2.d_data, 2 * ell, stream);

            // sum = v_folding[key_idx] × ginv_c2  (2×2*T_GSW) × (2*T_GSW×1) = 2×1
            DevicePolyMatrix sum = alloc_zero(2, 1, stream);
            launch_poly_mat_mul(sum.d_data,
                                v_gsw[key_idx].d_data, 2, 2 * ell,
                                ginv_c2.d_data, 1, stream);

            // sum += prod
            launch_poly_add_inplace(sum.d_data, prod.d_data, 2, 1, stream);

            // INTT(sum) → coefficient domain; count=2 for a 2×1 matrix
            launch_ntt_inverse(sum.d_data, 2, stream);
            cudaMemcpyAsync(v_cts[i].d_data, sum.d_data, sum.byte_size(),
                            cudaMemcpyDeviceToDevice, stream);

        }
    }
}

// ── pack_gpu ──────────────────────────────────────────────────────────────────

DevicePolyMatrix pack_gpu(const std::vector<DevicePolyMatrix>& v_ct,
                          const PublicParamsGPU& pp,
                          const SpiralParams& p, cudaStream_t stream)
{
    const uint32_t N      = SpiralParams::POLY_LEN;
    const uint32_t CRT    = SpiralParams::CRT_COUNT;
    const uint32_t n      = SpiralParams::N;      // 2
    const uint32_t T_CONV = SpiralParams::T_CONV; // 4

    assert(v_ct.size() >= static_cast<size_t>(n * n));
    assert(pp.v_packing.size() == n);

    // bits_per for gadget_invert with T_CONV limbs
    const uint32_t bits_per = get_bits_per(T_CONV);

    // result: (n+1) × n matrix in NTT domain
    DevicePolyMatrix result = alloc_zero(n + 1, n, stream);

    for (uint32_t c = 0; c < n; ++c) {
        // v_int: (n+1) × 1 NTT matrix
        DevicePolyMatrix v_int = alloc_zero(n + 1, 1, stream);

        for (uint32_t r = 0; r < n; ++r) {
            const DevicePolyMatrix& w  = pp.v_packing[r];  // (n+1) × T_CONV
            const DevicePolyMatrix& ct = v_ct[r * n + c];  // 2×1, coeff domain

            // ct_1 = ct.row(0)  (1×1 polynomial, coefficient domain)
            DevicePolyMatrix ct_1 = alloc_zero(1, 1, stream);
            cudaMemcpyAsync(ct_1.d_data, ct.d_data + 0 * CRT * N,
                            CRT * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);

            // ct_2 = ct.row(1)  → NTT domain
            DevicePolyMatrix ct_2 = alloc_zero(1, 1, stream);
            cudaMemcpyAsync(ct_2.d_data, ct.d_data + 1 * CRT * N,
                            CRT * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            launch_ntt_forward(ct_2.d_data, 1, stream);

            // gadget_invert(ct_1, rdim=1) → ginv  (T_CONV × 1, coeff domain)
            DevicePolyMatrix ginv = alloc_zero(T_CONV, 1, stream);
            launch_gadget_invert_crt(ginv.d_data, ct_1.d_data,
                                     1, 1, T_CONV, bits_per, stream);
            launch_ntt_forward(ginv.d_data, T_CONV, stream);

            // prod = w × ginv  ((n+1)×T_CONV) × (T_CONV×1) = (n+1)×1
            DevicePolyMatrix prod = alloc_zero(n + 1, 1, stream);
            launch_poly_mat_mul(prod.d_data, w.d_data, n + 1, T_CONV,
                                ginv.d_data, 1, stream);

            // v_int += prod
            launch_poly_add_inplace(v_int.d_data, prod.d_data, n + 1, 1, stream);

            // v_int.row(1+r) += ct_2  (add_into_at)
            launch_poly_add_inplace(v_int.d_data + (1 + r) * CRT * N,
                                    ct_2.d_data, 1, 1, stream);
        }

        // result.col(c) = v_int.col(0)
        for (uint32_t row = 0; row < n + 1; ++row) {
            const size_t src_off = static_cast<size_t>(row * 1 + 0) * CRT * N;
            const size_t dst_off = static_cast<size_t>(row * n + c) * CRT * N;
            cudaMemcpyAsync(result.d_data + dst_off, v_int.d_data + src_off,
                            CRT * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
        }
    }

    return result;
}
