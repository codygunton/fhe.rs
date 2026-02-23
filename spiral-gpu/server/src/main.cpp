// Spiral GPU PIR server — HTTP entry point.
//
// Same REST API as spiral-cpu/server:
//   POST /api/setup           → store public params, return UUID
//   POST /api/private-read    → PIR query, return response bytes
//   GET  /api/params          → JSON with spiral_params, setup_bytes, query_bytes, …
//   GET  /api/tile-mapping    → tile mapping JSON
//   GET  /api/metrics         → CPU/GPU utilization JSON
//
// Full implementation: Group F Task #5a.  This file will be expanded once
// Groups B–E kernels are complete.

#include <atomic>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

// POSIX mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "params.hpp"
#include "types.hpp"
#include "serialization.hpp"
#include "kernels/arith.cuh"
#include "kernels/ntt.cuh"

// Forward declarations from pipeline files
std::vector<uint8_t> process_query_gpu(
    const SpiralParams&, const PublicParamsGPU&,
    const uint8_t*, size_t, const DeviceDB&, cudaStream_t);
DeviceDB load_db_to_gpu(const uint8_t*, size_t, const SpiralParams&);

// ── CLI ───────────────────────────────────────────────────────────────────────
struct Config {
    std::string database;
    std::string tile_mapping;
    size_t      num_tiles   = 0;
    size_t      tile_size   = 20480;
    uint16_t    port        = 8082;
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --database <path> --tile-mapping <path> --num-tiles <N>\n"
        "          [--tile-size <B>] [--port <P>]\n",
        prog);
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { print_usage(argv[0]); std::exit(1); }
            return argv[i];
        };
        if      (arg == "--database")     cfg.database     = next();
        else if (arg == "--tile-mapping") cfg.tile_mapping = next();
        else if (arg == "--num-tiles")    cfg.num_tiles    = std::stoull(next());
        else if (arg == "--tile-size")    cfg.tile_size    = std::stoull(next());
        else if (arg == "--port")         cfg.port         = static_cast<uint16_t>(std::stoul(next()));
        else { print_usage(argv[0]); std::exit(1); }
    }
    if (cfg.database.empty() || cfg.tile_mapping.empty() || cfg.num_tiles == 0) {
        print_usage(argv[0]); std::exit(1);
    }
    return cfg;
}

// ── Server state ──────────────────────────────────────────────────────────────
struct ServerState {
    SpiralParams              params;
    DeviceDB                  db;
    std::string               tile_mapping_json;
    std::string               params_json;

    mutable std::shared_mutex sessions_mu;
    std::unordered_map<std::string, PublicParamsGPU> sessions;

    // Serialize GPU access (one query at a time for now)
    std::mutex gpu_mu;
};

// ── UUID generation ───────────────────────────────────────────────────────────
static std::string generate_uuid() {
    // Simple UUID v4 using /dev/urandom
    unsigned char buf[16];
    FILE* f = fopen("/dev/urandom", "rb");
    if (!f || fread(buf, 1, 16, f) != 16) {
        throw std::runtime_error("failed to read /dev/urandom");
    }
    if (f) fclose(f);
    buf[6] = (buf[6] & 0x0F) | 0x40;  // version 4
    buf[8] = (buf[8] & 0x3F) | 0x80;  // variant
    char uuid[37];
    snprintf(uuid, sizeof(uuid),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        buf[0],buf[1],buf[2],buf[3], buf[4],buf[5], buf[6],buf[7],
        buf[8],buf[9], buf[10],buf[11],buf[12],buf[13],buf[14],buf[15]);
    return uuid;
}

// ── Handlers ─────────────────────────────────────────────────────────────────

static void handle_setup(ServerState& st, const httplib::Request& req, httplib::Response& res) {
    const auto& body = req.body;
    try {
        auto pp = parse_public_params(
            reinterpret_cast<const uint8_t*>(body.data()), body.size(), st.params);
        std::string uuid = generate_uuid();
        {
            std::unique_lock lock(st.sessions_mu);
            st.sessions.emplace(uuid, std::move(pp));
        }
        std::cout << "[setup] session " << uuid << " stored\n";
        res.status = 200;
        res.set_content(uuid, "text/plain");
    } catch (const std::exception& e) {
        std::cerr << "[setup] error: " << e.what() << "\n";
        res.status = 400;
        res.set_content(e.what(), "text/plain");
    }
}

static void handle_private_read(ServerState& st, const httplib::Request& req,
                                 httplib::Response& res) {
    constexpr size_t UUID_LEN = 36;
    const auto& body = req.body;
    if (body.size() < UUID_LEN) {
        res.status = 400;
        res.set_content("body too short: need 36-byte UUID prefix", "text/plain");
        return;
    }
    std::string uuid(body.data(), UUID_LEN);
    const uint8_t* query_data = reinterpret_cast<const uint8_t*>(body.data()) + UUID_LEN;
    size_t query_len = body.size() - UUID_LEN;

    const PublicParamsGPU* pp_ptr = nullptr;
    {
        std::shared_lock lock(st.sessions_mu);
        auto it = st.sessions.find(uuid);
        if (it == st.sessions.end()) {
            res.status = 404;
            res.set_content("unknown session UUID: " + uuid, "text/plain");
            return;
        }
        pp_ptr = &it->second;
    }

    try {
        std::vector<uint8_t> response;
        {
            std::lock_guard gpu_lock(st.gpu_mu);
            response = process_query_gpu(st.params, *pp_ptr, query_data, query_len,
                                         st.db, /*stream=*/0);
        }
        res.status = 200;
        res.set_content(
            std::string(reinterpret_cast<const char*>(response.data()), response.size()),
            "application/octet-stream");
    } catch (const std::exception& e) {
        std::cerr << "[private-read] error: " << e.what() << "\n";
        res.status = 500;
        res.set_content("query processing failed", "text/plain");
    }
}

static void handle_params(ServerState& st, const httplib::Request&, httplib::Response& res) {
    nlohmann::json j;
    j["num_tiles"]    = st.params.num_items();
    j["tile_size"]    = st.params.db_item_size;
    j["spiral_params"]= st.params_json;
    j["setup_bytes"]  = st.params.setup_bytes();
    j["query_bytes"]  = st.params.query_bytes();
    j["num_items"]    = st.params.num_items();
    res.status = 200;
    res.set_content(j.dump(), "application/json");
}

static void handle_tile_mapping(ServerState& st, const httplib::Request&,
                                 httplib::Response& res) {
    res.status = 200;
    res.set_content(st.tile_mapping_json, "application/json");
}

static void handle_metrics(const httplib::Request&, httplib::Response& res) {
    // TODO: add real GPU utilization via NVML
    nlohmann::json j;
    j["cpu_percent"]     = 0;
    j["memory_used_mb"]  = 0;
    j["memory_total_mb"] = 0;
    res.status = 200;
    res.set_content(j.dump(), "application/json");
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    std::cout << "[spiral-gpu] Initializing...\n";

    // Initialize GPU math tables
    init_barrett_constants();
    init_ntt_tables();

    // Select Spiral params
    SpiralParams params = select_params(cfg.num_tiles, cfg.tile_size);
    std::string params_json = params.to_json();
    std::cout << "[spiral-gpu] params: " << params_json << "\n";
    std::cout << "[spiral-gpu] setup_bytes=" << params.setup_bytes()
              << " query_bytes=" << params.query_bytes()
              << " response_bytes=" << params.response_bytes() << "\n";

    // Load tile-mapping JSON
    std::string tile_mapping_json;
    {
        std::ifstream f(cfg.tile_mapping);
        if (!f) throw std::runtime_error("cannot open tile-mapping: " + cfg.tile_mapping);
        tile_mapping_json = std::string(
            std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // mmap database and load to GPU
    std::cout << "[spiral-gpu] Loading database: " << cfg.database << "\n";
    // Use POSIX mmap to avoid copying the file to a std::vector
    int fd = open(cfg.database.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open database: " + cfg.database);
    struct stat st_stat{};
    fstat(fd, &st_stat);
    const uint8_t* mapped = static_cast<const uint8_t*>(
        mmap(nullptr, st_stat.st_size, PROT_READ, MAP_SHARED, fd, 0));
    close(fd);
    if (mapped == MAP_FAILED) throw std::runtime_error("mmap failed");

    DeviceDB device_db = load_db_to_gpu(mapped, st_stat.st_size, params);
    munmap(const_cast<uint8_t*>(mapped), st_stat.st_size);
    std::cout << "[spiral-gpu] Database loaded ("
              << (device_db.byte_size() / 1024 / 1024) << " MB in VRAM)\n";

    // Build server state
    ServerState state;
    state.params           = params;
    state.db               = std::move(device_db);
    state.tile_mapping_json= tile_mapping_json;
    state.params_json      = params_json;

    // Start HTTP server
    httplib::Server svr;
    svr.set_payload_max_length(200 * 1024 * 1024);

    svr.Post("/api/setup", [&](const httplib::Request& req, httplib::Response& res) {
        handle_setup(state, req, res);
    });
    svr.Post("/api/private-read", [&](const httplib::Request& req, httplib::Response& res) {
        handle_private_read(state, req, res);
    });
    svr.Get("/api/params", [&](const httplib::Request& req, httplib::Response& res) {
        handle_params(state, req, res);
    });
    svr.Get("/api/tile-mapping", [&](const httplib::Request& req, httplib::Response& res) {
        handle_tile_mapping(state, req, res);
    });
    svr.Get("/api/metrics", [&](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res);
    });

    std::cout << "[spiral-gpu] Listening on port " << cfg.port << "\n";
    svr.listen("0.0.0.0", cfg.port);

    return 0;
}
