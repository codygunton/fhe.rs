#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "config.hpp"
#include "types.hpp"
#include "database/database_manager.hpp"
#include "pir/pir_engine.hpp"
#include "serialization/wire_format.hpp"

namespace mulpir::server {

/// PIR server that handles network connections and query processing.
///
/// This is the top-level component that:
/// - Loads and manages the database
/// - Accepts client connections
/// - Receives keys and queries
/// - Dispatches queries to the PIR engine
/// - Returns encrypted responses
class PIRServer {
public:
    /// Server configuration.
    struct Config {
        /// Path to the database file (binary tiles).
        std::string database_path;

        /// Number of tiles in the database.
        size_t num_tiles = 0;

        /// Size of each tile in bytes.
        size_t tile_size_bytes = 30720;  // 30 KiB

        /// Network port to listen on.
        uint16_t port = 8080;

        /// Maximum concurrent queries.
        size_t max_concurrent_queries = 4;

        /// Byte offset into the database file where tile data begins.
        /// Use 16 to skip a tiles.bin header (8-byte num_tiles + 8-byte tile_size).
        size_t data_offset = 0;

        /// Enable verbose logging.
        bool verbose = false;
    };

    /// Construct PIR server with configuration.
    ///
    /// @param config Server configuration.
    explicit PIRServer(const Config& config);

    ~PIRServer();

    // Disable copy and move
    PIRServer(const PIRServer&) = delete;
    PIRServer& operator=(const PIRServer&) = delete;
    PIRServer(PIRServer&&) = delete;
    PIRServer& operator=(PIRServer&&) = delete;

    /// Initialize the server: create context, load database.
    ///
    /// @throws std::runtime_error on failure.
    void initialize();

    /// Set client keys (received during setup phase).
    ///
    /// @param galois_key Serialized Galois key bytes.
    /// @param relin_key Serialized relinearization key bytes.
    void set_keys(
        std::span<const uint8_t> galois_key,
        std::span<const uint8_t> relin_key
    );

    /// Check if the server is ready to process queries.
    bool is_ready() const;

    /// Run the server (blocking).
    ///
    /// Listens for connections and processes queries until shutdown.
    void run();

    /// Shutdown the server.
    void shutdown();

    /// Process a single query (for testing/embedding).
    ///
    /// @param query PIR query.
    /// @return PIR response.
    PIRResponse handle_query(const PIRQuery& query);

    /// Get database size (number of tiles).
    size_t database_size() const;

    /// Get GPU memory usage in MB.
    size_t gpu_memory_mb() const;

    /// Get the last query statistics.
    const PIRStats& last_stats() const;

private:
    /// Accept incoming connections and enqueue work.
    void accept_loop();

    /// Processing thread: dequeue and process requests on GPU.
    void processing_loop();

    /// Handle a single client connection: read message and enqueue.
    ///
    /// @param client_fd Client socket file descriptor.
    void handle_connection(int client_fd);

    /// Process a received message.
    ///
    /// @param client_fd Client socket for sending response.
    /// @param msg_type Message type.
    /// @param payload Message payload.
    void process_message(
        int client_fd,
        serialization::WireFormat::MessageType msg_type,
        std::span<const uint8_t> payload
    );

    /// Send response to client.
    ///
    /// @param client_fd Client socket.
    /// @param status Response status.
    /// @param payload Response payload.
    void send_response(
        int client_fd,
        serialization::WireFormat::Status status,
        std::span<const uint8_t> payload
    );

    Config config_;
    HEContextPtr context_;
    std::unique_ptr<database::DatabaseManager> db_;
    std::unique_ptr<pir::PIREngine> engine_;
    std::unique_ptr<serialization::WireFormat> wire_;

    // Keys (set by client)
    std::unique_ptr<GaloisKey> galois_key_;
    std::unique_ptr<RelinKey> relin_key_;

    // Server state
    std::atomic<bool> running_{false};
    int server_fd_ = -1;
    PIRStats last_stats_;

    // Work queue for pipelined I/O + GPU processing
    struct PendingRequest {
        int client_fd;
        serialization::WireFormat::MessageType msg_type;
        std::vector<uint8_t> payload;
    };

    std::queue<PendingRequest> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread processing_thread_;
};

}  // namespace mulpir::server
