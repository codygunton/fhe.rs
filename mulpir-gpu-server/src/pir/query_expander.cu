#include "pir/query_expander.hpp"

#include <algorithm>
#include <stdexcept>

namespace mulpir::pir {

struct QueryExpander::Impl {
    HEContextPtr context;
    GaloisKey* galois_key;
    HEEncoder encoder;
    HEArithmeticOperator ops;

    /// Precomputed monomials: x^{-(n >> l)} for l in 0..level
    std::vector<Plaintext> monomials;

    Impl(
        HEContextPtr ctx,
        GaloisKey& gk
    )
        : context(ctx)
        , galois_key(&gk)
        , encoder(context)
        , ops(context, encoder) {
    }
};

QueryExpander::QueryExpander(
    HEContextPtr context,
    GaloisKey& galois_key,
    const PIRDimensions& dims
)
    : impl_(std::make_unique<Impl>(context, galois_key))
    , dims_(dims)
    , expansion_level_(dims.expansion_level) {
    precompute_monomials();
}

QueryExpander::~QueryExpander() = default;

QueryExpander::QueryExpander(QueryExpander&&) noexcept = default;
QueryExpander& QueryExpander::operator=(QueryExpander&&) noexcept = default;

size_t QueryExpander::expansion_level() const {
    return expansion_level_;
}

size_t QueryExpander::target_size() const {
    return dims_.dim1 + dims_.dim2;
}

void QueryExpander::precompute_monomials() {
    // Precompute monomials for the expansion algorithm.
    // monomial[l] = -x^{degree - 2^l} in the polynomial ring Z_t[x]/(x^N + 1)
    //
    // This matches fhe.rs: monomial[degree - (1 << l)] = -1
    // In unsigned mod-t representation, -1 is represented as (t - 1).

    const size_t degree = BFVConfig::POLY_DEGREE;
    impl_->monomials.resize(expansion_level_);

    for (size_t l = 0; l < expansion_level_; ++l) {
        // Position of the nonzero coefficient: degree - 2^l
        const size_t coeff_pos = degree - (1ULL << l);

        // Create coefficient vector with -1 (= t-1 mod t) at coeff_pos
        std::vector<uint64_t> coeffs(degree, 0);
        coeffs[coeff_pos] = BFVConfig::PLAINTEXT_MODULUS - 1;

        // Write polynomial coefficients directly, bypassing HEonGPU's batch
        // encoder. The MulPIR expansion algorithm requires actual polynomial
        // monomials (X^k), not batch-encoded slot vectors.
        std::vector<uint64_t> zeros(degree, 0);
        impl_->monomials[l] = Plaintext(impl_->context);
        impl_->encoder.encode(impl_->monomials[l], zeros);
        impl_->monomials[l].store_in_device();
        cudaMemcpy(impl_->monomials[l].data(), coeffs.data(),
                   degree * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
}

std::vector<Ciphertext> QueryExpander::expand(Ciphertext& query) {
    // Expansion algorithm from MulPIR/SealPIR
    //
    // Given: query ciphertext with encoded selection vector
    // Output: target_size ciphertexts, each encrypting 0 or 1
    //
    // Algorithm:
    //   out[0] = query
    //   for l in 0..level:
    //     step = 1 << l
    //     galois_elt = (degree >> l) + 1
    //     for i in 0..step:
    //       sub = apply_galois(out[i], galois_elt)
    //       if (step | i) < target_size:
    //         diff = out[i] - sub
    //         out[step | i] = diff * monomial[l]
    //       out[i] = out[i] + sub

    const size_t target = target_size();
    const size_t degree = BFVConfig::POLY_DEGREE;

    std::vector<Ciphertext> out(target);
    out[0] = std::move(query);

    for (size_t l = 0; l < expansion_level_; ++l) {
        const size_t step = 1ULL << l;
        const int galois_elt = static_cast<int>((degree >> l) + 1);

        for (size_t i = 0; i < step && i < target; ++i) {
            // Apply Galois automorphism: sub = galois(out[i], galois_elt)
            Ciphertext sub(impl_->context);
            impl_->ops.apply_galois(out[i], sub, *impl_->galois_key, galois_elt);

            const size_t new_idx = step | i;
            if (new_idx < target) {
                // diff = out[i] - sub
                Ciphertext diff(impl_->context);
                impl_->ops.sub(out[i], sub, diff);

                // out[new_idx] = diff * monomial[l]
                out[new_idx] = Ciphertext(impl_->context);
                impl_->ops.multiply_plain(diff, impl_->monomials[l], out[new_idx]);
            }

            // out[i] = out[i] + sub
            impl_->ops.add_inplace(out[i], sub);
        }
    }

    return out;
}

QueryExpander::ExpandedQuery QueryExpander::expand_split(Ciphertext& query) {
    auto all = expand(query);

    ExpandedQuery result;
    result.dim1_selection.reserve(dims_.dim1);
    result.dim2_selection.reserve(dims_.dim2);

    // First dim1 ciphertexts are for row selection
    for (size_t i = 0; i < dims_.dim1 && i < all.size(); ++i) {
        result.dim1_selection.push_back(std::move(all[i]));
    }

    // Remaining dim2 ciphertexts are for column selection
    for (size_t i = dims_.dim1; i < dims_.dim1 + dims_.dim2 && i < all.size(); ++i) {
        result.dim2_selection.push_back(std::move(all[i]));
    }

    return result;
}

}  // namespace mulpir::pir
