#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "config.hpp"
#include "types.hpp"
#include "database/database_manager.hpp"
#include "pir/query_expander.hpp"
#include "pir/dot_product.hpp"
#include "pir/selector.hpp"

namespace mulpir::pir {

/// Main PIR engine that orchestrates the complete query processing pipeline.
///
/// The pipeline consists of four phases:
/// 1. Expansion: Expand query ciphertext to dim1 + dim2 ciphertexts
/// 2. Dot Product: Compute first-dimension retrieval (CT-PT multiply+accumulate)
/// 3. Selection: Compute second-dimension retrieval (CT-CT multiply+accumulate)
/// 4. Finalization: Relinearize and compress the result
class PIREngine {
public:
    /// Construct PIR engine with all required components.
    ///
    /// @param context Shared HEonGPU BFV context.
    /// @param galois_key Galois key for query expansion.
    /// @param relin_key Relinearization key for result compression.
    /// @param database Reference to the loaded database.
    PIREngine(
        HEContextPtr context,
        GaloisKey& galois_key,
        RelinKey& relin_key,
        const database::DatabaseManager& database
    );

    ~PIREngine();

    // Disable copy, allow move
    PIREngine(const PIREngine&) = delete;
    PIREngine& operator=(const PIREngine&) = delete;
    PIREngine(PIREngine&&) noexcept;
    PIREngine& operator=(PIREngine&&) noexcept;

    /// Process a single PIR query.
    ///
    /// @param query Encrypted PIR query from client.
    /// @return Encrypted PIR response containing the requested element.
    PIRResponse process_query(const PIRQuery& query);

    /// Process multiple queries concurrently.
    ///
    /// Uses multiple CUDA streams for parallelism when enabled.
    ///
    /// @param queries Vector of encrypted PIR queries.
    /// @return Vector of encrypted PIR responses.
    std::vector<PIRResponse> process_queries(const std::vector<PIRQuery>& queries);

    /// Get statistics from the last processed query.
    const PIRStats& last_query_stats() const;

    /// Check if the engine is ready to process queries.
    bool is_ready() const;

    /// Get the database dimensions.
    const PIRDimensions& dimensions() const;

private:
    /// Process a single query and collect timing statistics.
    PIRResponse process_query_timed(const PIRQuery& query, PIRStats& stats);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<QueryExpander> expander_;
    std::unique_ptr<DotProduct> dot_product_;
    std::unique_ptr<Selector> selector_;
    const database::DatabaseManager* db_;
    PIRStats last_stats_;
};

}  // namespace mulpir::pir
