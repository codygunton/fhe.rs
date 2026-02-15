#include "pir/selector.hpp"

#include <stdexcept>

namespace mulpir::pir {

struct Selector::Impl {
    HEContextPtr context;
    RelinKey* relin_key;
    HEEncoder encoder;
    HEArithmeticOperator ops;

    Impl(
        HEContextPtr ctx,
        RelinKey& rk
    )
        : context(ctx)
        , relin_key(&rk)
        , encoder(context)
        , ops(context, encoder) {
    }
};

Selector::Selector(
    HEContextPtr context,
    RelinKey& relin_key
)
    : impl_(std::make_unique<Impl>(std::move(context), relin_key)) {
}

Selector::~Selector() = default;

Selector::Selector(Selector&&) noexcept = default;
Selector& Selector::operator=(Selector&&) noexcept = default;

Ciphertext Selector::select_and_relin(
    std::vector<Ciphertext>& partial_results,
    std::vector<Ciphertext>& selection_cts
) {
    if (partial_results.size() != selection_cts.size()) {
        throw std::invalid_argument(
            "Partial results and selection ciphertexts must have same size"
        );
    }
    if (partial_results.empty()) {
        throw std::invalid_argument("Cannot select from empty vectors");
    }

    // Compute: accumulator = sum(partial[i] * selection[i])
    //
    // Each CT-CT multiplication produces a 3-part ciphertext.
    // We accumulate all multiplications, then relinearize once at the end.
    //
    // Algorithm:
    //   accumulator = partial[0] * selection[0]
    //   for i in 1..n:
    //     temp = partial[i] * selection[i]
    //     accumulator += temp
    //   relinearize(accumulator)

    // First term
    Ciphertext accumulator(impl_->context);
    impl_->ops.multiply(partial_results[0], selection_cts[0], accumulator);

    // Remaining terms: multiply and accumulate
    for (size_t i = 1; i < partial_results.size(); ++i) {
        Ciphertext temp(impl_->context);
        impl_->ops.multiply(partial_results[i], selection_cts[i], temp);
        impl_->ops.add_inplace(accumulator, temp);
    }

    // Relinearize to bring back from 3-part to 2-part ciphertext
    impl_->ops.relinearize_inplace(accumulator, *impl_->relin_key);

    return accumulator;
}

void Selector::compress(Ciphertext& /*result*/) {
    // BFV in HEonGPU does not expose modulus switching.
    // The response is returned at the current modulus level.
    // This is a no-op for BFV.
}

Ciphertext Selector::select_relin_compress(
    std::vector<Ciphertext>& partial_results,
    std::vector<Ciphertext>& selection_cts
) {
    Ciphertext result = select_and_relin(partial_results, selection_cts);
    compress(result);
    return result;
}

}  // namespace mulpir::pir
