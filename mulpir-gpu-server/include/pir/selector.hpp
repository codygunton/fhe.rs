#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace mulpir::pir {

/// Performs second-dimension selection using ciphertext-ciphertext multiplication.
///
/// This is the second phase of MulPIR:
/// result = sum(partial[i] * selection[i]) for i in 0..dim2
///
/// After selection, the result is relinearized and optionally compressed
/// via modulus switching.
class Selector {
public:
    /// Construct selector with relinearization key.
    ///
    /// @param context Shared HEonGPU BFV context.
    /// @param relin_key Relinearization key for reducing ciphertext size.
    Selector(
        HEContextPtr context,
        RelinKey& relin_key
    );

    ~Selector();

    // Disable copy, allow move
    Selector(const Selector&) = delete;
    Selector& operator=(const Selector&) = delete;
    Selector(Selector&&) noexcept;
    Selector& operator=(Selector&&) noexcept;

    /// Select from partial results using encrypted selection vector.
    ///
    /// Computes: result = sum(partial_results[i] * selection_cts[i])
    /// Then relinearizes to reduce ciphertext to 2 polynomials.
    ///
    /// @param partial_results Results from first-dimension dot products (dim2 elements).
    /// @param selection_cts Second-dimension selection ciphertexts (dim2 elements).
    /// @return Relinearized result ciphertext.
    Ciphertext select_and_relin(
        std::vector<Ciphertext>& partial_results,
        std::vector<Ciphertext>& selection_cts
    );

    /// Apply modulus switching to compress the result.
    ///
    /// This reduces the size of the response ciphertext for transmission.
    ///
    /// @param result Ciphertext to compress (modified in place).
    void compress(Ciphertext& result);

    /// Select, relinearize, and compress in one operation.
    ///
    /// @param partial_results Results from first-dimension dot products.
    /// @param selection_cts Second-dimension selection ciphertexts.
    /// @return Compressed result ciphertext.
    Ciphertext select_relin_compress(
        std::vector<Ciphertext>& partial_results,
        std::vector<Ciphertext>& selection_cts
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mulpir::pir
