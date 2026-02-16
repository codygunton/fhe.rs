#include "server/pir_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mulpir::server {

PIRServer::PIRServer(const Config& config)
    : config_(config) {
}

PIRServer::~PIRServer() {
    shutdown();
}

void PIRServer::initialize() {
    std::cout << "Initializing MulPIR GPU Server..." << std::endl;

    // Create HEonGPU BFV context with parameters matching fhe.rs
    context_ = heongpu::GenHEContext<heongpu::Scheme::BFV>();
    context_->set_poly_modulus_degree(BFVConfig::POLY_DEGREE);
    context_->set_coeff_modulus_values(
        {BFVConfig::Q_MODULI[0], BFVConfig::Q_MODULI[1], BFVConfig::Q_MODULI[2]},
        {BFVConfig::P_MODULUS}
    );
    context_->set_plain_modulus(BFVConfig::PLAINTEXT_MODULUS);
    context_->generate();
    context_->print_parameters();

    // Create wire format handler
    wire_ = std::make_unique<serialization::WireFormat>(context_);

    // Create database manager
    db_ = std::make_unique<database::DatabaseManager>(context_, ServerConfig{});

    // Load database
    if (!config_.database_path.empty()) {
        std::cout << "Loading database from " << config_.database_path << "..." << std::endl;
        db_->load_database_mmap(
            config_.database_path,
            config_.num_tiles,
            config_.tile_size_bytes,
            config_.data_offset
        );
        std::cout << "  Loaded " << db_->num_tiles() << " tiles" << std::endl;
        std::cout << "  GPU memory: " << (db_->gpu_memory_used() / 1024 / 1024) << " MB" << std::endl;
    }

    std::cout << "Server initialized (waiting for keys)" << std::endl;
}

void PIRServer::set_keys(
    std::span<const uint8_t> galois_key_bytes,
    std::span<const uint8_t> relin_key_bytes
) {
    std::cout << "Setting client keys..." << std::endl;

    // Deserialize keys
    auto gk = wire_->deserialize_galois_key(galois_key_bytes);
    auto rk = wire_->deserialize_relin_key(relin_key_bytes);

    galois_key_ = std::make_unique<GaloisKey>(std::move(gk));
    relin_key_ = std::make_unique<RelinKey>(std::move(rk));

    if (db_ && db_->is_ready()) {
        // Create PIR engine
        engine_ = std::make_unique<pir::PIREngine>(
            context_, *galois_key_, *relin_key_, *db_
        );
    }

    std::cout << "Keys set, server ready for queries" << std::endl;
}

bool PIRServer::is_ready() const {
    return db_ && db_->is_ready() && engine_ && engine_->is_ready();
}

size_t PIRServer::database_size() const {
    return db_ ? db_->num_tiles() : 0;
}

size_t PIRServer::gpu_memory_mb() const {
    return db_ ? (db_->gpu_memory_used() / 1024 / 1024) : 0;
}

const PIRStats& PIRServer::last_stats() const {
    return last_stats_;
}

PIRResponse PIRServer::handle_query(const PIRQuery& query) {
    if (!is_ready()) {
        throw std::runtime_error("Server not ready");
    }

    auto response = engine_->process_query(query);
    last_stats_ = engine_->last_query_stats();
    return response;
}

void PIRServer::run() {
    // Create socket
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    // Set socket options
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Bind to port
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(config_.port);

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(server_fd_);
        server_fd_ = -1;
        throw std::runtime_error("Failed to bind to port " + std::to_string(config_.port));
    }

    // Listen
    if (listen(server_fd_, 5) < 0) {
        close(server_fd_);
        server_fd_ = -1;
        throw std::runtime_error("Failed to listen");
    }

    running_ = true;
    std::cout << "Server listening on port " << config_.port << std::endl;

    accept_loop();
}

void PIRServer::shutdown() {
    running_ = false;

    // Wake up processing thread
    queue_cv_.notify_all();

    // Close server socket to unblock accept()
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }

    // Wait for processing thread
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void PIRServer::accept_loop() {
    // Start the GPU processing thread
    processing_thread_ = std::thread([this]() { processing_loop(); });

    while (running_) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(
            server_fd_,
            reinterpret_cast<sockaddr*>(&client_addr),
            &client_len
        );

        if (client_fd < 0) {
            if (running_) {
                std::cerr << "Accept failed" << std::endl;
            }
            continue;
        }

        if (config_.verbose) {
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
            std::cout << "Connection from " << ip << std::endl;
        }

        handle_connection(client_fd);
    }
}

void PIRServer::processing_loop() {
    while (running_) {
        PendingRequest req;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !work_queue_.empty() || !running_;
            });

            if (!running_ && work_queue_.empty()) {
                break;
            }

            req = std::move(work_queue_.front());
            work_queue_.pop();
        }

        // Process the message on GPU thread
        process_message(req.client_fd, req.msg_type, req.payload);
        close(req.client_fd);
    }
}

void PIRServer::handle_connection(int client_fd) {
    // Read header (8 bytes)
    uint8_t header[8];
    ssize_t n = recv(client_fd, header, 8, MSG_WAITALL);
    if (n != 8) {
        close(client_fd);
        return;
    }

    // Parse header
    serialization::WireFormat::MessageType msg_type;
    uint32_t payload_length;
    if (!serialization::WireFormat::parse_header({header, 8}, msg_type, payload_length)) {
        send_response(client_fd, serialization::WireFormat::Status::ERROR_INVALID_MESSAGE, {});
        close(client_fd);
        return;
    }

    // Read payload
    std::vector<uint8_t> payload(payload_length);
    if (payload_length > 0) {
        n = recv(client_fd, payload.data(), payload_length, MSG_WAITALL);
        if (n != static_cast<ssize_t>(payload_length)) {
            close(client_fd);
            return;
        }
    }

    // Enqueue for processing thread (GPU operations must be serialized)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        work_queue_.push({client_fd, msg_type, std::move(payload)});
    }
    queue_cv_.notify_one();
}

void PIRServer::process_message(
    int client_fd,
    serialization::WireFormat::MessageType msg_type,
    std::span<const uint8_t> payload
) {
    using Status = serialization::WireFormat::Status;
    using MsgType = serialization::WireFormat::MessageType;

    try {
        switch (msg_type) {
            case MsgType::SET_GALOIS_KEY: {
                // Store Galois key (wait for relin key before creating engine)
                auto gk = wire_->deserialize_galois_key(payload);
                galois_key_ = std::make_unique<GaloisKey>(std::move(gk));
                send_response(client_fd, Status::OK, {});
                break;
            }

            case MsgType::SET_RELIN_KEY: {
                // Store relin key and create engine
                auto rk = wire_->deserialize_relin_key(payload);
                relin_key_ = std::make_unique<RelinKey>(std::move(rk));

                if (galois_key_ && db_ && db_->is_ready()) {
                    engine_ = std::make_unique<pir::PIREngine>(
                        context_, *galois_key_, *relin_key_, *db_
                    );
                }
                send_response(client_fd, Status::OK, {});
                break;
            }

            case MsgType::QUERY: {
                if (!is_ready()) {
                    send_response(client_fd, Status::ERROR_NOT_READY, {});
                    return;
                }

                // Deserialize query
                auto query = wire_->deserialize_query(payload);

                // Process query
                auto response = engine_->process_query(query);
                last_stats_ = engine_->last_query_stats();

                // Serialize and send response
                auto response_bytes = wire_->serialize_response(response);
                send_response(client_fd, Status::OK, response_bytes);
                break;
            }

            case MsgType::BATCH_QUERY: {
                if (!is_ready()) {
                    send_response(client_fd, Status::ERROR_NOT_READY, {});
                    return;
                }

                // Deserialize batch of queries
                auto queries = wire_->deserialize_batch_query(payload);
                std::cout << "Processing batch of " << queries.size() << " queries" << std::endl;

                // Process each query sequentially on GPU
                std::vector<PIRResponse> responses;
                responses.reserve(queries.size());
                for (auto& q : queries) {
                    responses.push_back(engine_->process_query(q));
                }
                last_stats_ = engine_->last_query_stats();

                // Serialize and send batch response
                auto response_bytes = wire_->serialize_batch_response(responses);
                send_response(client_fd, Status::OK, response_bytes);
                break;
            }

            default:
                send_response(client_fd, Status::ERROR_INVALID_MESSAGE, {});
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error processing message: " << e.what() << std::endl;
        send_response(client_fd, Status::ERROR_PROCESSING_FAILED, {});
    }
}

void PIRServer::send_response(
    int client_fd,
    serialization::WireFormat::Status status,
    std::span<const uint8_t> payload
) {
    // Build header
    auto header = serialization::WireFormat::build_response_header(
        status, static_cast<uint32_t>(payload.size())
    );

    // Send header
    send(client_fd, header.data(), header.size(), 0);

    // Send payload
    if (!payload.empty()) {
        send(client_fd, payload.data(), payload.size(), 0);
    }
}

}  // namespace mulpir::server
