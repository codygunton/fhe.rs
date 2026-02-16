#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "config.hpp"
#include "types.hpp"
#include "encoding/tile_encoder.hpp"

namespace mulpir::database {

/// Manages the PIR database, handling encoding, GPU upload, and access patterns.
///
/// The database is organized as a 2D matrix of plaintexts for efficient
/// MulPIR dot product operations. This class handles:
/// - Loading raw tiles from memory or files
/// - Encoding tiles to BFV plaintexts
/// - Uploading encoded data to GPU memory
/// - Providing column access for PIR dot products
class DatabaseManager {
public:
    /// Construct a database manager with the given context and configuration.
    ///
    /// @param context Shared HEonGPU BFV context.
    /// @param config Server configuration.
    DatabaseManager(
        HEContextPtr context,
        const ServerConfig& config
    );

    ~DatabaseManager();

    // Disable copy, allow move
    DatabaseManager(const DatabaseManager&) = delete;
    DatabaseManager& operator=(const DatabaseManager&) = delete;
    DatabaseManager(DatabaseManager&&) noexcept;
    DatabaseManager& operator=(DatabaseManager&&) noexcept;

    /// Load database from raw tile vectors.
    ///
    /// This encodes all tiles and uploads them to GPU memory.
    ///
    /// @param tiles Vector of raw tile data.
    void load_database(const std::vector<std::vector<uint8_t>>& tiles);

    /// Load database from a contiguous memory buffer.
    ///
    /// @param data Pointer to tile data (tiles are assumed to be contiguous).
    /// @param num_tiles Number of tiles in the buffer.
    /// @param tile_size Size of each tile in bytes.
    void load_database(const uint8_t* data, size_t num_tiles, size_t tile_size);

    /// Load database from a memory-mapped file.
    ///
    /// Useful for large databases that don't fit in RAM.
    ///
    /// @param path Path to the binary file containing tile data.
    /// @param num_tiles Number of tiles in the file.
    /// @param tile_size Size of each tile in bytes.
    /// @param data_offset Byte offset into the file where tile data begins (e.g. 16 to skip a header).
    void load_database_mmap(const std::string& path, size_t num_tiles, size_t tile_size,
                            size_t data_offset = 0);

    /// Check if the database is loaded and ready for queries.
    bool is_ready() const;

    /// Get the encoded database (GPU-resident).
    const EncodedDatabase& get_encoded() const;

    /// Get a column of the database matrix for dot product operations.
    ///
    /// In the 2D matrix layout, columns are accessed with stride dim2.
    /// Column j contains plaintexts at indices: j, j+dim2, j+2*dim2, ...
    ///
    /// @param column_index Column index (0 to dim2-1).
    /// @return Vector of pointers to plaintexts in this column.
    std::vector<const Plaintext*> get_column(size_t column_index) const;

    /// Get a specific plaintext by its matrix position.
    ///
    /// @param row Row index (0 to dim1-1).
    /// @param col Column index (0 to dim2-1).
    /// @return Pointer to the plaintext, or nullptr if out of bounds.
    const Plaintext* get_plaintext(size_t row, size_t col) const;

    /// Get the PIR dimensions of the loaded database.
    const PIRDimensions& dimensions() const;

    /// Get GPU memory used by the database (bytes).
    size_t gpu_memory_used() const;

    /// Get maximum number of tiles that can be stored.
    size_t max_supported_tiles() const;

    /// Get number of tiles currently loaded.
    size_t num_tiles() const;

    /// Clear the database and free GPU memory.
    void clear();

private:
    /// Reorganize plaintexts into the 2D matrix layout for PIR.
    ///
    /// The layout is row-major: db_[row * dim2 + col]
    void reorganize_for_pir();

    /// Upload plaintexts to GPU memory and transform to NTT.
    void upload_to_gpu();

    struct Impl;
    std::unique_ptr<Impl> impl_;
    ServerConfig config_;
    EncodedDatabase db_;
    std::unique_ptr<encoding::TileEncoder> encoder_;
    size_t num_tiles_ = 0;
    size_t tile_size_ = 0;
};

}  // namespace mulpir::database
