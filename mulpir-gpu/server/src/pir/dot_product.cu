#include "pir/dot_product.hpp"
#include "database/database_manager.hpp"

#include <stdexcept>

namespace mulpir::pir {

struct DotProduct::Impl {
    HEContextPtr context;
    HEEncoder encoder;
    HEArithmeticOperator ops;

    explicit Impl(HEContextPtr ctx)
        : context(ctx)
        , encoder(context)
        , ops(context, encoder) {
    }
};

DotProduct::DotProduct(
    HEContextPtr context
)
    : impl_(std::make_unique<Impl>(std::move(context))) {
}

DotProduct::~DotProduct() = default;

DotProduct::DotProduct(DotProduct&&) noexcept = default;
DotProduct& DotProduct::operator=(DotProduct&&) noexcept = default;

Ciphertext DotProduct::compute(
    std::vector<Ciphertext>& ciphertexts,
    std::vector<Plaintext*>& plaintexts
) {
    if (ciphertexts.size() != plaintexts.size()) {
        throw std::invalid_argument("Ciphertext and plaintext vectors must have same size");
    }
    if (ciphertexts.empty()) {
        throw std::invalid_argument("Cannot compute dot product of empty vectors");
    }

    // Compute: accumulator = sum(ct[i] * pt[i])
    //
    // Algorithm:
    //   accumulator = ct[0] * pt[0]
    //   for i in 1..n:
    //     temp = ct[i] * pt[i]
    //     accumulator += temp

    // First term
    Ciphertext accumulator(impl_->context);
    impl_->ops.multiply_plain(ciphertexts[0], *plaintexts[0], accumulator);

    // Remaining terms: multiply and accumulate
    for (size_t i = 1; i < ciphertexts.size(); ++i) {
        Ciphertext temp(impl_->context);
        impl_->ops.multiply_plain(ciphertexts[i], *plaintexts[i], temp);
        impl_->ops.add_inplace(accumulator, temp);
    }

    return accumulator;
}

std::vector<Ciphertext> DotProduct::compute_batch(
    std::vector<Ciphertext>& query_cts,
    const database::DatabaseManager& db
) {
    const auto& dims = db.dimensions();

    if (query_cts.size() != dims.dim1) {
        throw std::invalid_argument("Query ciphertexts must match dim1");
    }

    std::vector<Ciphertext> results;
    results.reserve(dims.dim2);

    for (size_t col = 0; col < dims.dim2; ++col) {
        // Get column of database plaintexts (non-const for HEonGPU API)
        auto column_ptrs = db.get_column(col);
        std::vector<Plaintext*> mut_ptrs;
        mut_ptrs.reserve(column_ptrs.size());
        for (auto* p : column_ptrs) {
            mut_ptrs.push_back(const_cast<Plaintext*>(p));
        }

        // Compute dot product of query ciphertexts with this column
        Ciphertext result = compute(query_cts, mut_ptrs);
        results.push_back(std::move(result));
    }

    return results;
}

}  // namespace mulpir::pir
