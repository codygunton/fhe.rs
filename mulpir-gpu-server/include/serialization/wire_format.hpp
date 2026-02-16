#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace mulpir::serialization {

/// Wire format for serializing PIR protocol messages.
///
/// This class handles conversion between C++ types and binary formats
/// suitable for network transmission. The format must be compatible
/// with fhe.rs client serialization.
class WireFormat {
public:
    /// Message types for the PIR protocol.
    enum class MessageType : uint32_t {
        SET_GALOIS_KEY = 0x01,
        SET_RELIN_KEY = 0x02,
        QUERY = 0x03,
        RESPONSE = 0x04,
        BATCH_QUERY = 0x05,
        ERROR = 0xFF,
    };

    /// Response status codes.
    enum class Status : uint32_t {
        OK = 0x00,
        ERROR_INVALID_MESSAGE = 0x01,
        ERROR_INVALID_QUERY = 0x02,
        ERROR_PROCESSING_FAILED = 0x03,
        ERROR_NOT_READY = 0x04,
    };

    /// Construct wire format handler.
    ///
    /// @param context Shared HEonGPU BFV context.
    explicit WireFormat(
        HEContextPtr context
    );

    ~WireFormat();

    // Disable copy, allow move
    WireFormat(const WireFormat&) = delete;
    WireFormat& operator=(const WireFormat&) = delete;
    WireFormat(WireFormat&&) noexcept;
    WireFormat& operator=(WireFormat&&) noexcept;

    /// Deserialize a PIR query from bytes.
    ///
    /// @param bytes Serialized query data.
    /// @return Deserialized PIR query.
    /// @throws std::runtime_error if deserialization fails.
    PIRQuery deserialize_query(std::span<const uint8_t> bytes);

    /// Serialize a PIR response to bytes.
    ///
    /// @param response PIR response to serialize.
    /// @return Serialized response data.
    std::vector<uint8_t> serialize_response(const PIRResponse& response);

    /// Deserialize Galois key from bytes.
    ///
    /// @param bytes Serialized key data.
    /// @return Deserialized Galois key.
    GaloisKey deserialize_galois_key(std::span<const uint8_t> bytes);

    /// Deserialize relinearization key from bytes.
    ///
    /// @param bytes Serialized key data.
    /// @return Deserialized relinearization key.
    RelinKey deserialize_relin_key(std::span<const uint8_t> bytes);

    /// Deserialize a batch of PIR queries from bytes.
    ///
    /// Wire format: [num_queries: u32 LE] [query_size: u32 LE] [query_bytes]...
    ///
    /// @param bytes Serialized batch query data.
    /// @return Vector of deserialized PIR queries.
    /// @throws std::runtime_error if deserialization fails.
    std::vector<PIRQuery> deserialize_batch_query(std::span<const uint8_t> bytes);

    /// Serialize a batch of PIR responses to bytes.
    ///
    /// Wire format: [num_responses: u32 LE] [response_size: u32 LE] [response_bytes]...
    ///
    /// @param responses Vector of PIR responses.
    /// @return Serialized batch response data.
    std::vector<uint8_t> serialize_batch_response(const std::vector<PIRResponse>& responses);

    /// Serialize a PIR query (for testing purposes).
    ///
    /// @param query PIR query to serialize.
    /// @return Serialized query data.
    std::vector<uint8_t> serialize_query(const PIRQuery& query);

    /// Build a message header.
    ///
    /// @param type Message type.
    /// @param payload_length Length of the payload.
    /// @return Header bytes (8 bytes: type + length).
    static std::vector<uint8_t> build_header(MessageType type, uint32_t payload_length);

    /// Build a response header.
    ///
    /// @param status Response status.
    /// @param payload_length Length of the payload.
    /// @return Header bytes (8 bytes: status + length).
    static std::vector<uint8_t> build_response_header(Status status, uint32_t payload_length);

    /// Parse a message header.
    ///
    /// @param header Header bytes (must be at least 8 bytes).
    /// @param type Output: message type.
    /// @param length Output: payload length.
    /// @return True if header is valid.
    static bool parse_header(
        std::span<const uint8_t> header,
        MessageType& type,
        uint32_t& length
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mulpir::serialization
