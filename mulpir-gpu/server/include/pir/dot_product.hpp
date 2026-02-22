#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "config.hpp"
#include "types.hpp"

// Forward declarations
namespace mulpir::database {
class DatabaseManager;
}

namespace mulpir::pir {

/// Computes dot products between ciphertext vectors and plaintext vectors.
///
/// This is the core operation for the first dimension of MulPIR:
/// result[j] = sum(query[i] * database[i, j]) for all columns j
class DotProduct {
public:
    /// Construct dot product operator.
    ///
    /// @param context Shared HEonGPU BFV context.
    explicit DotProduct(
        HEContextPtr context
    );

    ~DotProduct();

    // Disable copy, allow move
    DotProduct(const DotProduct&) = delete;
    DotProduct& operator=(const DotProduct&) = delete;
    DotProduct(DotProduct&&) noexcept;
    DotProduct& operator=(DotProduct&&) noexcept;

    /// Compute dot product of ciphertexts with plaintexts.
    ///
    /// Computes: sum(ct[i] * pt[i]) for i in 0..n
    ///
    /// @param ciphertexts Vector of query ciphertexts.
    /// @param plaintexts Vector of pointers to database plaintexts.
    /// @return Accumulated result ciphertext.
    Ciphertext compute(
        std::vector<Ciphertext>& ciphertexts,
        std::vector<Plaintext*>& plaintexts
    );

    /// Batch compute dot products for all columns.
    ///
    /// Computes dim2 dot products in parallel, one for each column.
    /// Each column's dot product uses all dim1 query ciphertexts.
    ///
    /// @param query_cts Query ciphertexts (dim1 elements).
    /// @param db Database manager providing column access.
    /// @return Vector of dim2 result ciphertexts.
    std::vector<Ciphertext> compute_batch(
        std::vector<Ciphertext>& query_cts,
        const database::DatabaseManager& db
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mulpir::pir
