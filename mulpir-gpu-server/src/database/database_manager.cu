#include "database/database_manager.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

// Memory-mapped file support
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mulpir::database {

struct DatabaseManager::Impl {
    HEContextPtr context;

    // Memory mapping state
    void* mmap_ptr = nullptr;
    size_t mmap_size = 0;
    int mmap_fd = -1;

    explicit Impl(HEContextPtr ctx)
        : context(std::move(ctx)) {
    }

    ~Impl() {
        unmap();
    }

    void unmap() {
#ifdef __linux__
        if (mmap_ptr != nullptr && mmap_ptr != MAP_FAILED) {
            munmap(mmap_ptr, mmap_size);
            mmap_ptr = nullptr;
            mmap_size = 0;
        }
        if (mmap_fd >= 0) {
            close(mmap_fd);
            mmap_fd = -1;
        }
#endif
    }
};

DatabaseManager::DatabaseManager(
    HEContextPtr context,
    const ServerConfig& config
)
    : impl_(std::make_unique<Impl>(context))
    , config_(config) {
    // Create encoder
    encoder_ = std::make_unique<encoding::TileEncoder>(impl_->context);
}

DatabaseManager::~DatabaseManager() {
    clear();
}

DatabaseManager::DatabaseManager(DatabaseManager&&) noexcept = default;
DatabaseManager& DatabaseManager::operator=(DatabaseManager&&) noexcept = default;

void DatabaseManager::load_database(const std::vector<std::vector<uint8_t>>& tiles) {
    if (tiles.empty()) {
        throw std::invalid_argument("Cannot load empty database");
    }

    num_tiles_ = tiles.size();
    tile_size_ = tiles[0].size();

    // Validate all tiles have same size
    for (const auto& tile : tiles) {
        if (tile.size() != tile_size_) {
            throw std::invalid_argument("All tiles must have the same size");
        }
    }

    // Compute PIR dimensions
    db_.dims = PIRDimensions::compute(num_tiles_, tile_size_);

    // Convert to spans for encoding
    std::vector<std::span<const uint8_t>> tile_spans;
    tile_spans.reserve(tiles.size());
    for (const auto& tile : tiles) {
        tile_spans.emplace_back(tile.data(), tile.size());
    }

    // Encode all tiles
    db_.plaintexts = encoder_->encode_tiles(tile_spans);

    // Reorganize for PIR access pattern
    reorganize_for_pir();

    // Upload to GPU
    upload_to_gpu();

    db_.is_ntt_form = true;
}

void DatabaseManager::load_database(const uint8_t* data, size_t num_tiles, size_t tile_size) {
    if (data == nullptr || num_tiles == 0 || tile_size == 0) {
        throw std::invalid_argument("Invalid database parameters");
    }

    num_tiles_ = num_tiles;
    tile_size_ = tile_size;

    // Compute PIR dimensions
    db_.dims = PIRDimensions::compute(num_tiles_, tile_size_);

    // Create spans for each tile
    std::vector<std::span<const uint8_t>> tile_spans;
    tile_spans.reserve(num_tiles);
    for (size_t i = 0; i < num_tiles; ++i) {
        tile_spans.emplace_back(data + i * tile_size, tile_size);
    }

    // Encode all tiles
    db_.plaintexts = encoder_->encode_tiles(tile_spans);

    // Reorganize for PIR access pattern
    reorganize_for_pir();

    // Upload to GPU
    upload_to_gpu();

    db_.is_ntt_form = true;
}

void DatabaseManager::load_database_mmap(
    const std::string& path,
    size_t num_tiles,
    size_t tile_size
) {
#ifdef __linux__
    // Open file
    impl_->mmap_fd = open(path.c_str(), O_RDONLY);
    if (impl_->mmap_fd < 0) {
        throw std::runtime_error("Failed to open database file: " + path);
    }

    // Get file size
    struct stat sb;
    if (fstat(impl_->mmap_fd, &sb) < 0) {
        close(impl_->mmap_fd);
        impl_->mmap_fd = -1;
        throw std::runtime_error("Failed to stat database file");
    }

    const size_t expected_size = num_tiles * tile_size;
    if (static_cast<size_t>(sb.st_size) < expected_size) {
        close(impl_->mmap_fd);
        impl_->mmap_fd = -1;
        throw std::runtime_error("Database file too small");
    }

    // Memory map the file
    impl_->mmap_size = expected_size;
    impl_->mmap_ptr = mmap(nullptr, impl_->mmap_size, PROT_READ, MAP_PRIVATE,
                           impl_->mmap_fd, 0);

    if (impl_->mmap_ptr == MAP_FAILED) {
        close(impl_->mmap_fd);
        impl_->mmap_fd = -1;
        impl_->mmap_ptr = nullptr;
        throw std::runtime_error("Failed to memory-map database file");
    }

    // Load from mapped memory
    load_database(static_cast<const uint8_t*>(impl_->mmap_ptr), num_tiles, tile_size);
#else
    // Fallback: read entire file into memory
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open database file: " + path);
    }

    std::vector<uint8_t> data(num_tiles * tile_size);
    file.read(reinterpret_cast<char*>(data.data()), data.size());

    if (!file) {
        throw std::runtime_error("Failed to read database file");
    }

    load_database(data.data(), num_tiles, tile_size);
#endif
}

bool DatabaseManager::is_ready() const {
    return db_.is_ready();
}

const EncodedDatabase& DatabaseManager::get_encoded() const {
    return db_;
}

std::vector<const Plaintext*> DatabaseManager::get_column(size_t column_index) const {
    if (!is_ready()) {
        throw std::runtime_error("Database not loaded");
    }
    if (column_index >= db_.dims.dim2) {
        throw std::out_of_range("Column index out of range");
    }

    std::vector<const Plaintext*> column;
    column.reserve(db_.dims.dim1);

    // Column j contains elements at positions: j, j+dim2, j+2*dim2, ...
    for (size_t row = 0; row < db_.dims.dim1; ++row) {
        const size_t idx = row * db_.dims.dim2 + column_index;
        if (idx < db_.plaintexts.size()) {
            column.push_back(&db_.plaintexts[idx]);
        }
    }

    return column;
}

const Plaintext* DatabaseManager::get_plaintext(size_t row, size_t col) const {
    if (!is_ready()) {
        return nullptr;
    }
    if (row >= db_.dims.dim1 || col >= db_.dims.dim2) {
        return nullptr;
    }

    const size_t idx = row * db_.dims.dim2 + col;
    if (idx >= db_.plaintexts.size()) {
        return nullptr;
    }

    return &db_.plaintexts[idx];
}

const PIRDimensions& DatabaseManager::dimensions() const {
    return db_.dims;
}

size_t DatabaseManager::gpu_memory_used() const {
    if (!is_ready()) {
        return 0;
    }

    // Each plaintext at level 1 has 2 moduli
    // Size = 2 * POLY_DEGREE * sizeof(uint64_t) per plaintext
    const size_t bytes_per_pt = 2 * BFVConfig::POLY_DEGREE * sizeof(uint64_t);
    return db_.plaintexts.size() * bytes_per_pt;
}

size_t DatabaseManager::max_supported_tiles() const {
    return config_.max_tiles_for_memory();
}

size_t DatabaseManager::num_tiles() const {
    return num_tiles_;
}

void DatabaseManager::clear() {
    db_.plaintexts.clear();
    db_.dims = PIRDimensions{};
    db_.is_ntt_form = false;
    num_tiles_ = 0;
    tile_size_ = 0;
    impl_->unmap();
}

void DatabaseManager::reorganize_for_pir() {
    // The encoding produces plaintexts in row order.
    // We may need to pad to fill the dim1 * dim2 matrix.

    const size_t target_size = db_.dims.dim1 * db_.dims.dim2;

    if (db_.plaintexts.size() < target_size) {
        // Pad with zero plaintexts using direct polynomial coefficients.
        // We initialize via the batch encoder (to set internal state),
        // then overwrite with zeros to ensure polynomial coefficient form.
        std::vector<uint64_t> zero_coeffs(BFVConfig::POLY_DEGREE, 0);
        mulpir::HEEncoder encoder(impl_->context);
        while (db_.plaintexts.size() < target_size) {
            Plaintext zero_pt(impl_->context);
            encoder.encode(zero_pt, zero_coeffs);
            zero_pt.store_in_device();
            cudaMemcpy(zero_pt.data(), zero_coeffs.data(),
                       BFVConfig::POLY_DEGREE * sizeof(uint64_t),
                       cudaMemcpyHostToDevice);
            db_.plaintexts.push_back(std::move(zero_pt));
        }
        cudaDeviceSynchronize();
    }

    // The current layout is already row-major (row * dim2 + col),
    // which is what we need for efficient column access.
}

void DatabaseManager::upload_to_gpu() {
    // Transfer all plaintexts to GPU memory
    for (auto& pt : db_.plaintexts) {
        pt.store_in_device();
    }
}

}  // namespace mulpir::database
