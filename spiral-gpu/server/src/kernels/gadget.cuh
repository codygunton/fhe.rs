#pragma once
// Gadget decomposition for Spiral PIR.
//
// Mirrors spiral-rs gadget.rs gadget_invert_rdim.

#include <cstdint>
#include <cuda_runtime.h>

// Gadget decomposition: extract num_elems limbs of bits_per bits each from
// each coefficient of the input matrix.
//
// d_out: output polynomial matrix, (rdim * num_elems) × cols, coefficient domain
// d_inp: input polynomial matrix,  rdim × cols, coefficient domain
// Both in modulus-major layout: poly[row,col] at (row*cols+col)*CRT*POLY_LEN.
//
// num_elems: number of limbs per decomposition (= out_rows / rdim)
// bits_per:  number of bits per limb
// stream:    CUDA stream
void launch_gadget_invert(uint64_t* d_out, const uint64_t* d_inp,
                          uint32_t rdim, uint32_t cols,
                          uint32_t num_elems, uint32_t bits_per,
                          cudaStream_t stream = 0);

// CRT-composing gadget decomposition.
// Same as launch_gadget_invert but first CRT-composes the two slots (mod q0,
// mod q1) back into the full 56-bit integer before extracting limbs, then
// writes the same small value to BOTH CRT output slots.
//
// Required for coefficient_expansion and fold_ciphertexts where the combined
// modulus (q0*q1 ≈ 2^56) matters for bit extraction, not just each residue.
void launch_gadget_invert_crt(uint64_t* d_out, const uint64_t* d_inp,
                               uint32_t rdim, uint32_t cols,
                               uint32_t num_elems, uint32_t bits_per,
                               cudaStream_t stream = 0);
