#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "config.hpp"
#include "types.hpp"
#include "encoding/tile_encoder.hpp"
#include "database/database_manager.hpp"
#include "pir/pir_engine.hpp"

using namespace mulpir;

/// Generate synthetic database for benchmarking.
std::vector<std::vector<uint8_t>> generate_synthetic_database(
    size_t num_tiles,
    size_t tile_size,
    uint64_t seed
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    std::vector<std::vector<uint8_t>> tiles;
    tiles.reserve(num_tiles);

    for (size_t i = 0; i < num_tiles; ++i) {
        std::vector<uint8_t> tile(tile_size);
        // First 4 bytes: tile index
        tile[0] = static_cast<uint8_t>(i & 0xFF);
        tile[1] = static_cast<uint8_t>((i >> 8) & 0xFF);
        tile[2] = static_cast<uint8_t>((i >> 16) & 0xFF);
        tile[3] = static_cast<uint8_t>((i >> 24) & 0xFF);
        // Rest: random data
        for (size_t j = 4; j < tile_size; ++j) {
            tile[j] = dist(rng);
        }
        tiles.push_back(std::move(tile));
    }

    return tiles;
}

/// Format time duration for display.
std::string format_time(double ms) {
    if (ms < 1.0) {
        return std::to_string(static_cast<int>(ms * 1000)) + " us";
    } else if (ms < 1000.0) {
        return std::to_string(static_cast<int>(ms * 10) / 10.0) + " ms";
    } else {
        return std::to_string(static_cast<int>(ms / 100) / 10.0) + " s";
    }
}

/// Print statistics summary.
void print_stats(const std::string& name, const std::vector<double>& samples) {
    if (samples.empty()) {
        return;
    }

    double sum = 0;
    double min_val = samples[0];
    double max_val = samples[0];

    for (double s : samples) {
        sum += s;
        min_val = std::min(min_val, s);
        max_val = std::max(max_val, s);
    }

    double mean = sum / samples.size();

    // Compute median
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];

    std::cout << std::setw(20) << name << ": "
              << "mean=" << std::setw(10) << format_time(mean)
              << "  median=" << std::setw(10) << format_time(median)
              << "  min=" << std::setw(10) << format_time(min_val)
              << "  max=" << std::setw(10) << format_time(max_val)
              << std::endl;
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --database <path>   Path to database file (optional)\n"
              << "  --num-tiles <n>     Number of tiles (default: 1000)\n"
              << "  --tile-size <n>     Tile size in bytes (default: 30720)\n"
              << "  --iterations <n>    Number of benchmark iterations (default: 10)\n"
              << "  --warmup <n>        Number of warmup iterations (default: 2)\n"
              << "  --seed <n>          Random seed (default: 12345)\n"
              << "  --help              Show this help\n";
}

/// Compute Galois elements needed for query expansion.
std::vector<uint32_t> expansion_galois_elements(size_t expansion_level) {
    std::vector<uint32_t> elts;
    elts.reserve(expansion_level);
    for (size_t l = 0; l < expansion_level; ++l) {
        elts.push_back(static_cast<uint32_t>((BFVConfig::POLY_DEGREE >> l) + 1));
    }
    return elts;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string database_path;
    size_t num_tiles = 1000;
    size_t tile_size = 30720;
    size_t iterations = 10;
    size_t warmup = 2;
    uint64_t seed = 12345;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--database" && i + 1 < argc) {
            database_path = argv[++i];
        } else if (arg == "--num-tiles" && i + 1 < argc) {
            num_tiles = std::stoul(argv[++i]);
        } else if (arg == "--tile-size" && i + 1 < argc) {
            tile_size = std::stoul(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoul(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoul(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== MulPIR GPU Benchmark ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Tiles: " << num_tiles << " x " << tile_size << " bytes" << std::endl;
    std::cout << "  Iterations: " << iterations << " (+ " << warmup << " warmup)" << std::endl;
    std::cout << std::endl;

    try {
        // Create BFV context
        std::cout << "Creating BFV context..." << std::endl;
        auto context = heongpu::GenHEContext<heongpu::Scheme::BFV>();
        context->set_poly_modulus_degree(BFVConfig::POLY_DEGREE);
        context->set_coeff_modulus_bit_sizes(
            {BFVConfig::MODULI_BITS[0], BFVConfig::MODULI_BITS[1], BFVConfig::MODULI_BITS[2]},
            {BFVConfig::P_MODULI_BITS[0]}
        );
        context->set_plain_modulus(BFVConfig::PLAINTEXT_MODULUS);
        context->generate();
        context->print_parameters();

        // Generate or load database
        std::cout << "Preparing database..." << std::endl;
        auto tiles = generate_synthetic_database(num_tiles, tile_size, seed);

        // Load database
        std::cout << "Loading database to GPU..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        database::DatabaseManager db(context, ServerConfig{});
        db.load_database(tiles);

        auto end = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "  Database loaded in " << format_time(load_ms) << std::endl;
        std::cout << "  GPU memory: " << (db.gpu_memory_used() / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Dimensions: " << db.dimensions().dim1 << " x "
                  << db.dimensions().dim2 << std::endl;
        std::cout << std::endl;

        // Benchmark encoding
        std::cout << "Benchmarking encoding..." << std::endl;
        std::vector<double> encode_times;
        encoding::TileEncoder encoder(context);

        for (size_t i = 0; i < warmup + iterations; ++i) {
            std::span<const uint8_t> tile_data(tiles[i % num_tiles].data(), tile_size);

            start = std::chrono::high_resolution_clock::now();
            auto encoded = encoder.encode_tile(tile_data);
            end = std::chrono::high_resolution_clock::now();

            if (i >= warmup) {
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                encode_times.push_back(ms);
            }
        }

        // Generate keys for PIR benchmarks
        std::cout << "Generating keys..." << std::endl;
        const auto& dims = db.dimensions();

        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);

        heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey<heongpu::Scheme::BFV> relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        auto galois_elts = expansion_galois_elements(dims.expansion_level);
        heongpu::Galoiskey<heongpu::Scheme::BFV> galois_key(context, galois_elts);
        keygen.generate_galois_key(galois_key, secret_key);

        std::cout << "Keys generated." << std::endl;

        // Create PIR engine
        pir::PIREngine engine(context, galois_key, relin_key, db);

        // Create encryptor for query generation
        heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context, public_key);
        mulpir::HEEncoder he_encoder(context);

        // Benchmark PIR queries
        std::cout << "Benchmarking PIR queries..." << std::endl;
        std::vector<double> expand_times;
        std::vector<double> dot_product_times;
        std::vector<double> selection_times;
        std::vector<double> total_times;

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<size_t> index_dist(0, num_tiles - 1);

        for (size_t i = 0; i < warmup + iterations; ++i) {
            // Generate random query
            size_t query_index = index_dist(rng);
            size_t row = query_index / dims.dim2;
            size_t col = query_index % dims.dim2;

            std::vector<uint64_t> query_coeffs(BFVConfig::POLY_DEGREE, 0);
            query_coeffs[row] = 1;
            query_coeffs[dims.dim1 + col] = 1;

            Plaintext query_pt(context);
            he_encoder.encode(query_pt, query_coeffs);

            Ciphertext query_ct(context);
            encryptor.encrypt(query_ct, query_pt);

            PIRQuery query;
            query.encrypted_query = std::move(query_ct);
            query.metadata.query_id = query_index;

            // Process query
            auto response = engine.process_query(query);
            const auto& stats = engine.last_query_stats();

            if (i >= warmup) {
                expand_times.push_back(stats.expansion_ms);
                dot_product_times.push_back(stats.dot_product_ms);
                selection_times.push_back(stats.selection_ms);
                total_times.push_back(stats.total_ms);
            }
        }

        std::cout << std::endl;
        std::cout << "=== Results ===" << std::endl;
        print_stats("Tile encoding", encode_times);
        print_stats("Query expansion", expand_times);
        print_stats("Dot product", dot_product_times);
        print_stats("Selection+relin", selection_times);
        print_stats("Total query", total_times);

        if (!total_times.empty()) {
            double mean_total = 0;
            for (double t : total_times) mean_total += t;
            mean_total /= total_times.size();
            std::cout << std::endl;
            std::cout << "  Throughput: " << std::setprecision(2) << std::fixed
                      << (1000.0 / mean_total) << " queries/sec" << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Benchmark complete." << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
