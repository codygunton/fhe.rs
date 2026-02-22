#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace mulpir::pir {

/// Expands a single PIR query ciphertext into dim1 + dim2 ciphertexts.
///
/// The expansion uses Galois automorphisms to obliviously expand the
/// encoded selection vector. This is the first phase of MulPIR processing.
class QueryExpander {
public:
    /// Construct query expander with the required keys.
    ///
    /// @param context Shared HEonGPU BFV context.
    /// @param galois_key Galois key for rotations/automorphisms.
    /// @param dims PIR dimensions for the database.
    QueryExpander(
        HEContextPtr context,
        GaloisKey& galois_key,
        const PIRDimensions& dims
    );

    ~QueryExpander();

    // Disable copy, allow move
    QueryExpander(const QueryExpander&) = delete;
    QueryExpander& operator=(const QueryExpander&) = delete;
    QueryExpander(QueryExpander&&) noexcept;
    QueryExpander& operator=(QueryExpander&&) noexcept;

    /// Expanded query split into two parts.
    struct ExpandedQuery {
        /// Selection ciphertexts for first dimension (indices 0 to dim1-1).
        std::vector<Ciphertext> dim1_selection;

        /// Selection ciphertexts for second dimension (indices dim1 to dim1+dim2-1).
        std::vector<Ciphertext> dim2_selection;

        /// Total number of ciphertexts.
        size_t total_size() const {
            return dim1_selection.size() + dim2_selection.size();
        }
    };

    /// Expand query ciphertext to dim1 + dim2 ciphertexts.
    ///
    /// @param query Encrypted PIR query.
    /// @return Vector of all expanded ciphertexts.
    std::vector<Ciphertext> expand(Ciphertext& query);

    /// Expand and split query into dim1 and dim2 components.
    ///
    /// This is the preferred interface for PIR processing as it
    /// provides the separation needed for subsequent operations.
    ///
    /// @param query Encrypted PIR query.
    /// @return ExpandedQuery with dim1 and dim2 selections.
    ExpandedQuery expand_split(Ciphertext& query);

    /// Get the expansion level (log2 of target size rounded up).
    size_t expansion_level() const;

    /// Get the target expansion size (dim1 + dim2).
    size_t target_size() const;

private:
    /// Precompute monomials for expansion algorithm.
    ///
    /// Monomials are: x^{-(n >> l)} for l in 0..level
    /// where n is the polynomial degree.
    void precompute_monomials();



    struct Impl;
    std::unique_ptr<Impl> impl_;
    PIRDimensions dims_;
    size_t expansion_level_;
};

}  // namespace mulpir::pir
