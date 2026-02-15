#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <heongpu/heongpu.hpp>

#include "config.hpp"

namespace mulpir {

constexpr auto HE_SCHEME = heongpu::Scheme::BFV;

// Type aliases for HEonGPU types
// Note: heongpu::HEContext<S> is already std::shared_ptr<HEContextImpl<S>>
using HEContextPtr = heongpu::HEContext<HE_SCHEME>;
using Ciphertext = heongpu::Ciphertext<HE_SCHEME>;
using Plaintext = heongpu::Plaintext<HE_SCHEME>;
using SecretKey = heongpu::Secretkey<HE_SCHEME>;
using PublicKey = heongpu::Publickey<HE_SCHEME>;
using GaloisKey = heongpu::Galoiskey<HE_SCHEME>;
using RelinKey = heongpu::Relinkey<HE_SCHEME>;
using HEEncoder = heongpu::HEEncoder<HE_SCHEME>;
using HEEncryptor = heongpu::HEEncryptor<HE_SCHEME>;
using HEDecryptor = heongpu::HEDecryptor<HE_SCHEME>;
using HEKeyGenerator = heongpu::HEKeyGenerator<HE_SCHEME>;
using HEArithmeticOperator = heongpu::HEArithmeticOperator<HE_SCHEME>;

/// PIR database dimensions computed from database size and BFV parameters.
struct PIRDimensions {
    /// Total number of elements in the database.
    size_t num_elements = 0;

    /// Number of database elements that fit in one plaintext.
    size_t elements_per_plaintext = 0;

    /// Number of rows in the encoded database (after packing).
    size_t num_rows = 0;

    /// First dimension size: ceil(sqrt(num_rows)).
    size_t dim1 = 0;

    /// Second dimension size: ceil(num_rows / dim1).
    size_t dim2 = 0;

    /// Expansion level: ceil(log2(dim1 + dim2)).
    /// This determines how many levels of Galois key expansion are needed.
    size_t expansion_level = 0;

    /// Compute PIR dimensions for a given database configuration.
    static PIRDimensions compute(size_t db_size, size_t element_size, const BFVConfig& config = {}) {
        PIRDimensions dims;
        dims.num_elements = db_size;

        // Compute elements per plaintext (at least 1 — large elements span multiple plaintexts)
        const size_t bits_per_coeff = config.BITS_PER_COEFF;
        const size_t bytes_per_plaintext = (bits_per_coeff * config.POLY_DEGREE) / 8;
        dims.elements_per_plaintext = std::max<size_t>(1, bytes_per_plaintext / element_size);

        // Compute number of rows (plaintexts needed)
        dims.num_rows = (db_size + dims.elements_per_plaintext - 1) / dims.elements_per_plaintext;

        // Compute 2D dimensions for MulPIR
        dims.dim1 = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(dims.num_rows))));
        dims.dim2 = (dims.num_rows + dims.dim1 - 1) / dims.dim1;

        // Expansion level = ceil(log2(dim1 + dim2))
        const size_t total_dims = dims.dim1 + dims.dim2;
        size_t power_of_two = 1;
        dims.expansion_level = 0;
        while (power_of_two < total_dims) {
            power_of_two *= 2;
            dims.expansion_level++;
        }

        return dims;
    }

    bool is_valid() const {
        return num_elements > 0 && elements_per_plaintext > 0 &&
               num_rows > 0 && dim1 > 0 && dim2 > 0 && expansion_level > 0;
    }

    size_t total_plaintexts() const {
        return dim1 * dim2;
    }
};

/// Encoded database ready for PIR operations.
struct EncodedDatabase {
    /// Plaintexts stored in row-major order.
    std::vector<Plaintext> plaintexts;

    /// Database dimensions.
    PIRDimensions dims;

    /// Whether plaintexts are in NTT form (ready for GPU operations).
    bool is_ntt_form = false;

    bool is_ready() const {
        return !plaintexts.empty() && dims.is_valid() && is_ntt_form;
    }
};

/// A PIR query from the client.
struct PIRQuery {
    /// Encrypted query ciphertext.
    Ciphertext encrypted_query;

    /// Query metadata for validation.
    struct Metadata {
        size_t expected_db_size = 0;
        size_t expected_element_size = 0;
        uint64_t query_id = 0;
    } metadata;
};

/// A PIR response from the server.
struct PIRResponse {
    /// Encrypted result ciphertext.
    Ciphertext encrypted_result;

    /// Response metadata.
    struct Metadata {
        uint64_t query_id = 0;
        double processing_time_ms = 0.0;
        bool is_compressed = true;
    } metadata;
};

/// Statistics from PIR query processing.
struct PIRStats {
    double expansion_ms = 0.0;
    double dot_product_ms = 0.0;
    double selection_ms = 0.0;
    double relinearization_ms = 0.0;
    double mod_switch_ms = 0.0;
    double total_ms = 0.0;
    size_t ciphertexts_processed = 0;

    double sum_stages() const {
        return expansion_ms + dot_product_ms + selection_ms +
               relinearization_ms + mod_switch_ms;
    }
};

/// Result of a batch query operation.
struct BatchQueryResult {
    std::vector<PIRResponse> responses;
    PIRStats stats;
    size_t num_successful = 0;
    size_t num_failed = 0;
};

}  // namespace mulpir
