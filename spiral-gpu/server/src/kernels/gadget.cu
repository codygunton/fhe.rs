// Gadget decomposition kernel — STUB
//
// Full implementation: Group B Task #1d.
// Mirrors spiral-rs gadget.rs gadget_invert_rdim.

#include "params.hpp"
#include "arith.cuh"

#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

// Gadget decomposition: extract num_elems limbs from each coefficient.
// d_out: (rdim * num_elems) × cols polynomial matrix
// d_inp: rdim × cols polynomial matrix
// Both in coefficient domain.
void launch_gadget_invert(uint64_t* /*d_out*/, const uint64_t* /*d_inp*/,
                          uint32_t /*rdim*/, uint32_t /*cols*/,
                          uint32_t /*num_elems*/, uint32_t /*bits_per*/,
                          cudaStream_t /*stream*/) {
    throw std::runtime_error("gadget_invert not yet implemented (Group B Task #1d)");
}
