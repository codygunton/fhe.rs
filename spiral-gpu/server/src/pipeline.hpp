#pragma once
// Forward declarations for Spiral PIR pipeline functions.
// Included by process_query.cu and any other file that calls these functions.

#include "types.hpp"
#include "params.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

// Load and preprocess the database from raw tile bytes, upload to GPU.
DeviceDB load_db_to_gpu(const uint8_t* raw_tiles, size_t file_size,
                        const SpiralParams& p);

// Expand a single query ciphertext into (v_reg_reoriented, v_folding).
// v_reg_reoriented: CRT-packed reoriented flat buffer (dim0 * 2 * POLY_LEN u64s)
// v_folding:        nu_2 GSW ciphertexts (2 × 2*T_GSW each)
std::pair<std::vector<uint64_t>, std::vector<DevicePolyMatrix>>
expand_query_gpu(const CiphertextGPU& ct, const PublicParamsGPU& pp,
                 const SpiralParams& p, cudaStream_t stream = 0);

// First-dimension database multiply.
// d_out must point to num_per pre-allocated DevicePolyMatrix (each 2×1).
void db_multiply_gpu(DevicePolyMatrix* d_out,
                     const uint64_t* d_firstdim,
                     const DeviceDB& db,
                     uint32_t instance, uint32_t trial,
                     const SpiralParams& p, cudaStream_t stream = 0);

// Batched first-dimension database multiply.
// Computes all B queries × num_per dot products in one kernel launch (one DB scan).
//
// d_batch_out: caller-allocated flat buffer of B * num_per * 2 * CRT * N u64s.
//   Output polynomial (b, ii, row) is at offset (b*num_per*2 + ii*2 + row)*CRT*N,
//   suitable for launch_ntt_inverse(d_batch_out, B*num_per*2, stream).
//
// d_firstdim_batch: flat buffer of B * dim0 * 2 * N u64s.
//   Query b's data starts at b * dim0*2*N.
void db_multiply_batch_gpu(uint64_t* d_batch_out,
                            const uint64_t* d_firstdim_batch,
                            const DeviceDB& db,
                            uint32_t batch_size,
                            uint32_t instance, uint32_t trial,
                            const SpiralParams& p, cudaStream_t stream = 0);

// Second-dimension fold: reduces v_cts (num_per → 1) in-place.
void fold_ciphertexts_gpu(std::vector<DevicePolyMatrix>& v_cts,
                          const std::vector<DevicePolyMatrix>& v_gsw,
                          const SpiralParams& p, cudaStream_t stream = 0);

// Pack N×N ciphertexts into one (N+1)×N result.
DevicePolyMatrix pack_gpu(const std::vector<DevicePolyMatrix>& v_ct,
                          const PublicParamsGPU& pp,
                          const SpiralParams& p, cudaStream_t stream = 0);

// Process B queries as a batch — shares one DB scan per (instance, trial).
// queries: vector of (pointer, length) pairs, one per query.
// Returns one response byte vector per query, in the same order.
std::vector<std::vector<uint8_t>> process_queries_batch_gpu(
    const SpiralParams& p,
    const PublicParamsGPU& pp,
    const std::vector<std::pair<const uint8_t*, size_t>>& queries,
    const DeviceDB& db,
    cudaStream_t stream = 0);
