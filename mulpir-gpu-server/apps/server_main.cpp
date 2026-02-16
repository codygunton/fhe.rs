#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "server/pir_server.hpp"

// Global server pointer for signal handling
static mulpir::server::PIRServer* g_server = nullptr;

void signal_handler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->shutdown();
    }
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --database <path>   Path to database file (required)\n"
              << "  --num-tiles <n>     Number of tiles in database (required)\n"
              << "  --tile-size <n>     Size of each tile in bytes (default: 30720)\n"
              << "  --data-offset <n>   Byte offset to tile data in file (default: 0, use 16 to skip tiles.bin header)\n"
              << "  --port <n>          Server port (default: 8080)\n"
              << "  --verbose           Enable verbose output\n"
              << "  --help              Show this help message\n"
              << "\n"
              << "Example:\n"
              << "  " << program << " --database tiles.bin --num-tiles 100000 --port 8080\n";
}

int main(int argc, char** argv) {
    mulpir::server::PIRServer::Config config;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--database" && i + 1 < argc) {
            config.database_path = argv[++i];
        } else if (arg == "--num-tiles" && i + 1 < argc) {
            config.num_tiles = std::stoul(argv[++i]);
        } else if (arg == "--tile-size" && i + 1 < argc) {
            config.tile_size_bytes = std::stoul(argv[++i]);
        } else if (arg == "--data-offset" && i + 1 < argc) {
            config.data_offset = std::stoul(argv[++i]);
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = static_cast<uint16_t>(std::stoul(argv[++i]));
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (config.database_path.empty()) {
        std::cerr << "Error: --database is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (config.num_tiles == 0) {
        std::cerr << "Error: --num-tiles is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Print configuration
    std::cout << "=== MulPIR GPU Server ===" << std::endl;
    std::cout << "Database: " << config.database_path << std::endl;
    std::cout << "Tiles: " << config.num_tiles << " x " << config.tile_size_bytes << " bytes" << std::endl;
    std::cout << "Port: " << config.port << std::endl;
    std::cout << std::endl;

    try {
        // Create and initialize server
        mulpir::server::PIRServer server(config);
        g_server = &server;

        // Set up signal handlers
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Initialize
        server.initialize();

        std::cout << std::endl;
        std::cout << "Database loaded: " << server.database_size() << " tiles" << std::endl;
        std::cout << "GPU memory used: " << server.gpu_memory_mb() << " MB" << std::endl;
        std::cout << std::endl;
        std::cout << "Server ready, waiting for client keys..." << std::endl;
        std::cout << "Press Ctrl+C to shutdown" << std::endl;
        std::cout << std::endl;

        // Run server (blocks until shutdown)
        server.run();

        g_server = nullptr;
        std::cout << "Server stopped" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
