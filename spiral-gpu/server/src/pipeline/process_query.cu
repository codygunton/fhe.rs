// Top-level query processing pipeline for Spiral PIR.
//
// Mirrors spiral-rs server.rs process_query() exactly.
// Pipeline:
//   1. parse_query → CiphertextGPU
//   2. expand_query_gpu → (v_reg_reoriented, v_folding)
//   3. Upload v_reg_reoriented to GPU
//   4. For each instance:
//      a. For each trial (n*n trials):
//         - db_multiply_gpu → num_per intermediate 2×1 NTT ciphertexts
//         - INTT each intermediate → coefficient domain
//         - fold_ciphertexts_gpu → 1 coefficient-domain ciphertext
//         - Collect 1 ciphertext per trial
//      b. pack_gpu(v_ct[instance]) → (N+1)×N packed NTT ciphertext
//   5. encode_response → byte vector

#include "types.hpp"
#include "params.hpp"
#include "serialization.hpp"
#include "pipeline.hpp"
#include "kernels/ntt.cuh"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

// ── process_queries_batch_gpu ─────────────────────────────────────────────────
//
// Process B Spiral PIR queries as a batch, sharing one DB scan per trial.
// Returns one response byte vector per query, in the same order as `queries`.
//
// For each (instance, trial) pair, a single db_multiply_batch_gpu call produces
// all B × num_per intermediates in one pass over the database, reducing DRAM
// bandwidth from O(B * DB_size) to O(DB_size).
std::vector<std::vector<uint8_t>> process_queries_batch_gpu(
    const SpiralParams& p,
    const PublicParamsGPU& pp,
    const std::vector<std::pair<const uint8_t*, size_t>>& queries,
    const DeviceDB& db,
    cudaStream_t stream)
{
    const uint32_t N       = SpiralParams::POLY_LEN;
    const uint32_t CRT     = SpiralParams::CRT_COUNT;
    const uint32_t n       = SpiralParams::N;
    const uint32_t trials  = n * n;
    const uint32_t num_per = p.num_per();
    const uint32_t B       = static_cast<uint32_t>(queries.size());

    // Step 1: Parse and expand all B queries on GPU
    std::vector<std::vector<uint64_t>> firstdims(B);
    std::vector<std::vector<DevicePolyMatrix>> foldings(B);
    for (uint32_t b = 0; b < B; ++b) {
        CiphertextGPU ct = parse_query(queries[b].first, queries[b].second, p);
        auto [fd, fld] = expand_query_gpu(ct, pp, p, stream);
        firstdims[b] = std::move(fd);
        foldings[b]  = std::move(fld);
    }

    // Step 2: Stack all firstdims into d_firstdim_batch on device
    const size_t firstdim_words = static_cast<size_t>(p.dim0()) * 2 * N;
    uint64_t* d_firstdim_batch = nullptr;
    {
        cudaError_t err = cudaMallocAsync(&d_firstdim_batch,
                                          B * firstdim_words * sizeof(uint64_t), stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("process_queries_batch_gpu: cudaMallocAsync d_firstdim_batch: ")
                + cudaGetErrorString(err));
        }
    }
    for (uint32_t b = 0; b < B; ++b) {
        cudaMemcpyAsync(d_firstdim_batch + b * firstdim_words,
                        firstdims[b].data(), firstdim_words * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);
    }

    // Step 3: Allocate reusable batch output buffer
    // Layout: polynomial (b, ii, row) at (b*num_per*2 + ii*2 + row)*CRT*N
    const size_t ct_words        = static_cast<size_t>(2) * CRT * N;  // one 2×1 matrix
    const size_t batch_out_words = static_cast<size_t>(B) * num_per * 2 * CRT * N;
    uint64_t* d_batch_out = nullptr;
    {
        cudaError_t err = cudaMallocAsync(&d_batch_out,
                                          batch_out_words * sizeof(uint64_t), stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(d_firstdim_batch, stream);
            throw std::runtime_error(
                std::string("process_queries_batch_gpu: cudaMallocAsync d_batch_out: ")
                + cudaGetErrorString(err));
        }
    }

    // Step 4: For each instance and trial, batch-multiply then per-query fold
    // packed_per_query[b] accumulates one packed matrix per instance for query b
    std::vector<std::vector<DevicePolyMatrix>> packed_per_query(B);
    for (uint32_t b = 0; b < B; ++b) packed_per_query[b].reserve(p.instances);

    for (uint32_t instance = 0; instance < p.instances; ++instance) {
        // Collect one folded 2×1 ciphertext per trial per query
        std::vector<std::vector<DevicePolyMatrix>> v_ct_per_query(B);
        for (uint32_t b = 0; b < B; ++b) v_ct_per_query[b].reserve(trials);

        for (uint32_t trial = 0; trial < trials; ++trial) {
            // Zero batch output buffer before multiply
            cudaMemsetAsync(d_batch_out, 0, batch_out_words * sizeof(uint64_t), stream);

            // ONE batch DB multiply: all B queries, this (instance, trial)
            db_multiply_batch_gpu(d_batch_out, d_firstdim_batch, db,
                                  B, instance, trial, p, stream);

            // Batch INTT: treat buffer as B*num_per*2 separate polynomials
            launch_ntt_inverse(d_batch_out, B * num_per * 2, stream);

            // Per-query fold using non-owning views into d_batch_out
            for (uint32_t b = 0; b < B; ++b) {
                // Create view for each of the num_per intermediates for query b
                std::vector<DevicePolyMatrix> intermediate;
                intermediate.reserve(num_per);
                for (uint32_t k = 0; k < num_per; ++k) {
                    uint64_t* ptr = d_batch_out
                                    + static_cast<size_t>(b * num_per + k) * ct_words;
                    intermediate.push_back(DevicePolyMatrix::view(ptr, 2, 1));
                }

                // fold_ciphertexts_gpu reduces intermediate in-place via the views
                fold_ciphertexts_gpu(intermediate, foldings[b], p, stream);

                // Copy the folded result from d_batch_out to an owning allocation
                DevicePolyMatrix owned(2, 1);
                cudaMemcpyAsync(owned.d_data, intermediate[0].d_data,
                                ct_words * sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, stream);
                v_ct_per_query[b].push_back(std::move(owned));
            }
        }

        // Pack each query's trial ciphertexts for this instance
        for (uint32_t b = 0; b < B; ++b) {
            packed_per_query[b].push_back(pack_gpu(v_ct_per_query[b], pp, p, stream));
        }
    }

    // Step 5: Free working buffers, sync, then encode all responses
    cudaFreeAsync(d_firstdim_batch, stream);
    cudaFreeAsync(d_batch_out, stream);
    cudaStreamSynchronize(stream);

    std::vector<std::vector<uint8_t>> responses;
    responses.reserve(B);
    for (uint32_t b = 0; b < B; ++b) {
        responses.push_back(encode_response(packed_per_query[b], p));
    }
    return responses;
}

// ── process_query_gpu ─────────────────────────────────────────────────────────

// Process one Spiral PIR query end-to-end on GPU.
// Returns response bytes identical to spiral-rs process_query().
std::vector<uint8_t> process_query_gpu(
    const SpiralParams& p,
    const PublicParamsGPU& pp,
    const uint8_t* query_bytes_raw,
    size_t query_len,
    const DeviceDB& db,
    cudaStream_t stream)
{
    const uint32_t N       = SpiralParams::POLY_LEN;
    const uint32_t n       = SpiralParams::N;
    const uint32_t trials  = n * n;
    const uint32_t dim0    = p.dim0();
    const uint32_t num_per = p.num_per();

    // 1. Parse query
    CiphertextGPU ct = parse_query(query_bytes_raw, query_len, p);

    // 2. Expand query: coefficient_expansion + regev_to_gsw
    auto [v_reg_reoriented, v_folding] = expand_query_gpu(ct, pp, p, stream);

    // 3. Upload v_reg_reoriented to GPU
    const size_t reg_words = static_cast<size_t>(dim0) * 2 * N;
    uint64_t* d_firstdim = nullptr;
    cudaError_t err = cudaMallocAsync(&d_firstdim, reg_words * sizeof(uint64_t), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("process_query_gpu: cudaMallocAsync d_firstdim: ") +
                                 cudaGetErrorString(err));
    }
    cudaMemcpyAsync(d_firstdim, v_reg_reoriented.data(), reg_words * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, stream);

    // 4. For each instance, run db_multiply + fold + pack
    std::vector<DevicePolyMatrix> result_mats;
    result_mats.reserve(p.instances);

    for (uint32_t instance = 0; instance < p.instances; ++instance) {
        // Collect one ciphertext per trial (n*n = 4)
        std::vector<DevicePolyMatrix> v_ct;
        v_ct.reserve(trials);

        for (uint32_t trial = 0; trial < trials; ++trial) {
            // Allocate num_per intermediate 2×1 NTT output ciphertexts
            std::vector<DevicePolyMatrix> intermediate;
            intermediate.reserve(num_per);
            for (uint32_t k = 0; k < num_per; ++k) {
                intermediate.emplace_back(2, 1);
                cudaMemsetAsync(intermediate.back().d_data, 0,
                                intermediate.back().byte_size(), stream);
            }

            // db_multiply: output is in NTT domain
            db_multiply_gpu(intermediate.data(), d_firstdim, db,
                            instance, trial, p, stream);

            // INTT each intermediate → coefficient domain for folding
            // count=2 because each intermediate is a 2×1 matrix (2 polynomials)
            for (uint32_t k = 0; k < num_per; ++k) {
                launch_ntt_inverse(intermediate[k].d_data, 2, stream);
            }

            // fold_ciphertexts: reduces num_per → 1 in-place
            fold_ciphertexts_gpu(intermediate, v_folding, p, stream);

            // intermediate[0] is now the single result (coefficient domain)
            v_ct.push_back(std::move(intermediate[0]));
        }

        // pack: (N+1)×N NTT packed ciphertext
        DevicePolyMatrix packed = pack_gpu(v_ct, pp, p, stream);
        result_mats.push_back(std::move(packed));
    }

    cudaFreeAsync(d_firstdim, stream);
    cudaStreamSynchronize(stream);

    // 5. Encode response
    return encode_response(result_mats, p);
}
