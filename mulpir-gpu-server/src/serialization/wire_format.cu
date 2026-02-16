#include "serialization/wire_format.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mulpir::serialization {

struct WireFormat::Impl {
    HEContextPtr context;

    explicit Impl(HEContextPtr ctx)
        : context(std::move(ctx)) {
    }
};

WireFormat::WireFormat(
    HEContextPtr context
)
    : impl_(std::make_unique<Impl>(std::move(context))) {
}

WireFormat::~WireFormat() = default;

WireFormat::WireFormat(WireFormat&&) noexcept = default;
WireFormat& WireFormat::operator=(WireFormat&&) noexcept = default;

std::vector<uint8_t> WireFormat::build_header(MessageType type, uint32_t payload_length) {
    std::vector<uint8_t> header(8);

    const auto type_val = static_cast<uint32_t>(type);
    std::memcpy(header.data(), &type_val, 4);
    std::memcpy(header.data() + 4, &payload_length, 4);

    return header;
}

std::vector<uint8_t> WireFormat::build_response_header(Status status, uint32_t payload_length) {
    std::vector<uint8_t> header(8);

    const auto status_val = static_cast<uint32_t>(status);
    std::memcpy(header.data(), &status_val, 4);
    std::memcpy(header.data() + 4, &payload_length, 4);

    return header;
}

bool WireFormat::parse_header(
    std::span<const uint8_t> header,
    MessageType& type,
    uint32_t& length
) {
    if (header.size() < 8) {
        return false;
    }

    uint32_t type_val;
    std::memcpy(&type_val, header.data(), 4);
    std::memcpy(&length, header.data() + 4, 4);

    switch (type_val) {
        case static_cast<uint32_t>(MessageType::SET_GALOIS_KEY):
        case static_cast<uint32_t>(MessageType::SET_RELIN_KEY):
        case static_cast<uint32_t>(MessageType::QUERY):
        case static_cast<uint32_t>(MessageType::RESPONSE):
        case static_cast<uint32_t>(MessageType::BATCH_QUERY):
        case static_cast<uint32_t>(MessageType::ERROR):
            type = static_cast<MessageType>(type_val);
            return true;
        default:
            return false;
    }
}

PIRQuery WireFormat::deserialize_query(std::span<const uint8_t> bytes) {
    if (bytes.empty()) {
        throw std::runtime_error("Cannot deserialize empty query");
    }

    PIRQuery query;
    std::string buf(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::istringstream stream(buf, std::ios::binary);
    query.encrypted_query.load(stream);

    return query;
}

std::vector<uint8_t> WireFormat::serialize_response(const PIRResponse& response) {
    std::ostringstream stream(std::ios::binary);
    const_cast<Ciphertext&>(response.encrypted_result).save(stream);
    auto s = stream.str();
    return {s.begin(), s.end()};
}

GaloisKey WireFormat::deserialize_galois_key(std::span<const uint8_t> bytes) {
    if (bytes.empty()) {
        throw std::runtime_error("Cannot deserialize empty Galois key");
    }

    GaloisKey key;
    std::string buf(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::istringstream stream(buf, std::ios::binary);
    key.load(stream);
    // Set context after loading for proper GPU memory management
    key.set_context(impl_->context);
    return key;
}

RelinKey WireFormat::deserialize_relin_key(std::span<const uint8_t> bytes) {
    if (bytes.empty()) {
        throw std::runtime_error("Cannot deserialize empty relinearization key");
    }

    RelinKey key;
    std::string buf(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::istringstream stream(buf, std::ios::binary);
    key.load(stream);
    // Set context after loading for proper GPU memory management
    key.set_context(impl_->context);
    return key;
}

std::vector<PIRQuery> WireFormat::deserialize_batch_query(std::span<const uint8_t> bytes) {
    if (bytes.size() < 4) {
        throw std::runtime_error("Batch query too short for header");
    }

    uint32_t num_queries;
    std::memcpy(&num_queries, bytes.data(), 4);

    if (num_queries == 0 || num_queries > 64) {
        throw std::runtime_error("Invalid batch query count: " + std::to_string(num_queries));
    }

    constexpr uint32_t MAX_QUERY_SIZE = 10 * 1024 * 1024; // 10 MB per query

    std::vector<PIRQuery> queries;
    queries.reserve(num_queries);
    size_t offset = 4;

    for (uint32_t i = 0; i < num_queries; ++i) {
        if (offset + 4 > bytes.size()) {
            throw std::runtime_error("Batch query truncated at query " + std::to_string(i));
        }

        uint32_t query_size;
        std::memcpy(&query_size, bytes.data() + offset, 4);
        offset += 4;

        if (query_size == 0 || query_size > MAX_QUERY_SIZE) {
            throw std::runtime_error("Invalid query size: " + std::to_string(query_size));
        }

        if (offset + query_size > bytes.size()) {
            throw std::runtime_error("Batch query data truncated at query " + std::to_string(i));
        }

        queries.push_back(deserialize_query({bytes.data() + offset, query_size}));
        offset += query_size;
    }

    return queries;
}

std::vector<uint8_t> WireFormat::serialize_batch_response(const std::vector<PIRResponse>& responses) {
    // First serialize all individual responses to get sizes
    std::vector<std::vector<uint8_t>> serialized;
    serialized.reserve(responses.size());
    size_t total_size = 4; // num_responses header

    for (const auto& resp : responses) {
        auto data = serialize_response(resp);
        total_size += 4 + data.size(); // size prefix + data
        serialized.push_back(std::move(data));
    }

    std::vector<uint8_t> result;
    result.reserve(total_size);

    // Write num_responses
    auto num = static_cast<uint32_t>(responses.size());
    result.resize(4);
    std::memcpy(result.data(), &num, 4);

    // Write each response with size prefix
    for (const auto& data : serialized) {
        auto sz = static_cast<uint32_t>(data.size());
        size_t pos = result.size();
        result.resize(pos + 4);
        std::memcpy(result.data() + pos, &sz, 4);
        result.insert(result.end(), data.begin(), data.end());
    }

    return result;
}

std::vector<uint8_t> WireFormat::serialize_query(const PIRQuery& query) {
    std::ostringstream stream(std::ios::binary);
    const_cast<Ciphertext&>(query.encrypted_query).save(stream);
    auto s = stream.str();
    return {s.begin(), s.end()};
}

}  // namespace mulpir::serialization
