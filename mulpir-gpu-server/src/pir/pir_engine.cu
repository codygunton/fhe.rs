#include "pir/pir_engine.hpp"

#include <chrono>
#include <stdexcept>

namespace mulpir::pir {

struct PIREngine::Impl {
    HEContextPtr context;

    explicit Impl(HEContextPtr ctx)
        : context(std::move(ctx)) {
    }
};

PIREngine::PIREngine(
    HEContextPtr context,
    GaloisKey& galois_key,
    RelinKey& relin_key,
    const database::DatabaseManager& database
)
    : impl_(std::make_unique<Impl>(context))
    , db_(&database) {
    if (!database.is_ready()) {
        throw std::invalid_argument("Database must be loaded before creating PIR engine");
    }

    // Create sub-components
    expander_ = std::make_unique<QueryExpander>(
        context, galois_key, database.dimensions()
    );

    dot_product_ = std::make_unique<DotProduct>(context);

    selector_ = std::make_unique<Selector>(context, relin_key);
}

PIREngine::~PIREngine() = default;

PIREngine::PIREngine(PIREngine&&) noexcept = default;
PIREngine& PIREngine::operator=(PIREngine&&) noexcept = default;

bool PIREngine::is_ready() const {
    return db_ != nullptr && db_->is_ready() &&
           expander_ != nullptr && dot_product_ != nullptr && selector_ != nullptr;
}

const PIRDimensions& PIREngine::dimensions() const {
    return db_->dimensions();
}

const PIRStats& PIREngine::last_query_stats() const {
    return last_stats_;
}

PIRResponse PIREngine::process_query(const PIRQuery& query) {
    PIRStats stats;
    auto response = process_query_timed(query, stats);
    last_stats_ = stats;
    return response;
}

PIRResponse PIREngine::process_query_timed(const PIRQuery& query, PIRStats& stats) {
    if (!is_ready()) {
        throw std::runtime_error("PIR engine not ready");
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // Phase 1: Query Expansion
    // Expand single query ciphertext to dim1 + dim2 ciphertexts
    auto expand_start = std::chrono::high_resolution_clock::now();
    auto query_ct = const_cast<Ciphertext&>(query.encrypted_query);
    auto expanded = expander_->expand_split(query_ct);
    auto expand_end = std::chrono::high_resolution_clock::now();
    stats.expansion_ms = std::chrono::duration<double, std::milli>(expand_end - expand_start).count();
    stats.ciphertexts_processed = expanded.total_size();

    // Phase 2: First-Dimension Dot Products
    // Compute CT-PT dot products for all columns
    auto dot_start = std::chrono::high_resolution_clock::now();
    auto partial_results = dot_product_->compute_batch(
        expanded.dim1_selection, *db_
    );
    auto dot_end = std::chrono::high_resolution_clock::now();
    stats.dot_product_ms = std::chrono::duration<double, std::milli>(dot_end - dot_start).count();

    // Phase 3: Second-Dimension Selection + Relinearization
    // Compute CT-CT multiplication, accumulation, and relinearize
    auto sel_start = std::chrono::high_resolution_clock::now();
    auto result = selector_->select_and_relin(
        partial_results, expanded.dim2_selection
    );
    auto sel_end = std::chrono::high_resolution_clock::now();
    stats.selection_ms = std::chrono::duration<double, std::milli>(sel_end - sel_start).count();

    // No compression phase - BFV in HEonGPU does not support modulus switching

    auto total_end = std::chrono::high_resolution_clock::now();
    stats.total_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start
    ).count();

    // Build response
    PIRResponse response;
    response.encrypted_result = std::move(result);
    response.metadata.query_id = query.metadata.query_id;
    response.metadata.processing_time_ms = stats.total_ms;
    response.metadata.is_compressed = false;

    return response;
}

std::vector<PIRResponse> PIREngine::process_queries(
    const std::vector<PIRQuery>& queries
) {
    std::vector<PIRResponse> responses;
    responses.reserve(queries.size());

    PIRStats aggregate_stats{};

    for (const auto& query : queries) {
        PIRStats query_stats;
        auto response = process_query_timed(query, query_stats);
        responses.push_back(std::move(response));

        // Accumulate statistics
        aggregate_stats.expansion_ms += query_stats.expansion_ms;
        aggregate_stats.dot_product_ms += query_stats.dot_product_ms;
        aggregate_stats.selection_ms += query_stats.selection_ms;
        aggregate_stats.relinearization_ms += query_stats.relinearization_ms;
        aggregate_stats.total_ms += query_stats.total_ms;
        aggregate_stats.ciphertexts_processed += query_stats.ciphertexts_processed;
    }

    last_stats_ = aggregate_stats;
    return responses;
}

}  // namespace mulpir::pir
