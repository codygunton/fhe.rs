//! Generates test vectors for MulPIR GPU server validation.
//!
//! This utility creates a deterministic set of test data including:
//! - Synthetic database tiles
//! - BFV keys (Galois and relinearization)
//! - Encrypted PIR queries for specific indices
//! - Expected plaintext results
//!
//! Run with: cargo run --example generate_test_vectors -- --output-dir ./test_vectors
#![expect(missing_docs, reason = "examples/benches/tests omit docs by design")]
#![expect(
    clippy::indexing_slicing,
    reason = "performance or example code relies on validated indices"
)]

mod util;

use anyhow::{Context, Result};
use clap::Parser;
use fhe::bfv::{
    self, BfvParameters, BfvParametersBuilder, Ciphertext, EvaluationKey, EvaluationKeyBuilder,
    Plaintext, RelinearizationKey, SecretKey,
};
use fhe_traits::{FheEncoder, FheEncrypter, Serialize};
use fhe_util::{inverse, transcode_from_bytes};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Serialize as SerdeSerialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};
use util::number_elements_per_plaintext;

/// Command-line arguments for the test vector generator.
#[derive(Parser)]
#[command(name = "generate_test_vectors")]
#[command(about = "Generate test vectors for MulPIR GPU server validation")]
struct Args {
    /// Output directory for test vectors.
    #[arg(long, default_value = "./test_vectors")]
    output_dir: PathBuf,

    /// Number of synthetic tiles to generate.
    #[arg(long, default_value = "100")]
    num_tiles: usize,

    /// Size of each tile in bytes.
    #[arg(long, default_value = "30720")]
    tile_size: usize,

    /// Random seed for deterministic generation.
    #[arg(long, default_value = "12345")]
    seed: u64,

    /// Indices to generate queries for (comma-separated).
    /// If not specified, uses: 0, 1, dim1, dim1*dim2-1, 42
    #[arg(long, value_delimiter = ',')]
    query_indices: Option<Vec<usize>>,
}

/// BFV parameters matching the GPU server configuration.
#[derive(SerdeSerialize)]
struct ParamsJson {
    poly_degree: usize,
    plaintext_modulus: u64,
    moduli_bits: Vec<usize>,
    num_tiles: usize,
    tile_size: usize,
    elements_per_plaintext: usize,
    num_rows: usize,
    dim1: usize,
    dim2: usize,
    expansion_level: usize,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("MulPIR Test Vector Generator");
    info!("Output: {:?}", args.output_dir);
    info!("Tiles: {} x {} bytes", args.num_tiles, args.tile_size);
    info!("Seed: {}", args.seed);

    // BFV parameters matching the GPU server
    let degree = 8192;
    let plaintext_modulus: u64 = (1 << 20) + (1 << 19) + (1 << 17) + (1 << 16) + (1 << 14) + 1;
    let moduli_sizes = [50, 55, 55];

    // Validate parameters
    let max_element_size = ((plaintext_modulus.ilog2() as usize) * degree) / 8;
    if args.tile_size > max_element_size {
        anyhow::bail!(
            "Tile size {} exceeds maximum {} for these BFV parameters",
            args.tile_size,
            max_element_size
        );
    }

    // Initialize deterministic RNG
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);

    // Generate synthetic tiles (deterministic based on seed)
    info!("Generating {} synthetic tiles...", args.num_tiles);
    let tiles = generate_synthetic_tiles(args.num_tiles, args.tile_size, &mut rng);

    // Build BFV parameters
    info!("Building BFV parameters...");
    let params = BfvParametersBuilder::new()
        .set_degree(degree)
        .set_plaintext_modulus(plaintext_modulus)
        .set_moduli_sizes(&moduli_sizes)
        .build_arc()
        .context("Failed to build BFV parameters")?;

    // Compute PIR dimensions
    let plaintext_nbits = plaintext_modulus.ilog2() as usize;
    let elements_per_plaintext =
        number_elements_per_plaintext(degree, plaintext_nbits, args.tile_size);
    let num_rows = args.num_tiles.div_ceil(elements_per_plaintext);
    let dim1 = (num_rows as f64).sqrt().ceil() as usize;
    let dim2 = num_rows.div_ceil(dim1);
    let expansion_level = (dim1 + dim2).next_power_of_two().ilog2() as usize;

    info!("PIR dimensions: {}x{} (expansion level {})", dim1, dim2, expansion_level);

    // Generate keys
    info!("Generating keys...");
    let sk = SecretKey::random(&params, &mut rng);

    let ek = EvaluationKeyBuilder::new_leveled(&sk, 1, 0)
        .context("Failed to create evaluation key builder")?
        .enable_expansion(expansion_level)
        .context("Failed to enable expansion")?
        .build(&mut rng)
        .context("Failed to build evaluation key")?;

    let rk = RelinearizationKey::new_leveled(&sk, 1, 1, &mut rng)
        .context("Failed to build relinearization key")?;

    // Determine query indices
    let query_indices: Vec<usize> = args.query_indices.unwrap_or_else(|| {
        let mut indices = vec![0, 1];
        if dim1 < args.num_tiles {
            indices.push(dim1);
        }
        if dim1 * dim2 > 1 && dim1 * dim2 - 1 < args.num_tiles {
            indices.push(dim1 * dim2 - 1);
        }
        if 42 < args.num_tiles && !indices.contains(&42) {
            indices.push(42);
        }
        indices.sort();
        indices.dedup();
        indices
    });

    info!("Query indices: {:?}", query_indices);

    // Create output directories
    let output_dir = &args.output_dir;
    fs::create_dir_all(output_dir)?;
    fs::create_dir_all(output_dir.join("keys"))?;
    fs::create_dir_all(output_dir.join("queries"))?;
    fs::create_dir_all(output_dir.join("expected"))?;

    // Save parameters
    let params_json = ParamsJson {
        poly_degree: degree,
        plaintext_modulus,
        moduli_bits: moduli_sizes.to_vec(),
        num_tiles: args.num_tiles,
        tile_size: args.tile_size,
        elements_per_plaintext,
        num_rows,
        dim1,
        dim2,
        expansion_level,
    };
    let params_file = File::create(output_dir.join("params.json"))?;
    serde_json::to_writer_pretty(params_file, &params_json)?;
    info!("Saved params.json");

    // Save raw tiles
    let tiles_path = output_dir.join("tiles.bin");
    let mut tiles_file = File::create(&tiles_path)?;
    for tile in &tiles {
        tiles_file.write_all(tile)?;
    }
    info!("Saved tiles.bin ({} bytes)", tiles.len() * args.tile_size);

    // Save keys
    let ek_bytes = ek.to_bytes();
    fs::write(output_dir.join("keys/galois.bin"), &ek_bytes)?;
    info!("Saved galois.bin ({} bytes)", ek_bytes.len());

    let rk_bytes = rk.to_bytes();
    fs::write(output_dir.join("keys/relin.bin"), &rk_bytes)?;
    info!("Saved relin.bin ({} bytes)", rk_bytes.len());

    // Save secret key (for testing only - would not be shared in production)
    let sk_bytes = sk.to_bytes();
    fs::write(output_dir.join("keys/secret.bin"), &sk_bytes)?;
    info!("Saved secret.bin ({} bytes) [TEST ONLY]", sk_bytes.len());

    // Generate and save queries
    for &index in &query_indices {
        if index >= args.num_tiles {
            warn!("Skipping index {} (out of range)", index);
            continue;
        }

        info!("Generating query for index {}...", index);

        // Create the query
        let query = create_pir_query(
            index,
            args.tile_size,
            &params,
            &sk,
            dim1,
            dim2,
            elements_per_plaintext,
            &mut rng,
        )?;

        // Save query ciphertext
        let query_bytes = query.to_bytes();
        fs::write(
            output_dir.join(format!("queries/query_{}.bin", index)),
            &query_bytes,
        )?;
        info!("  Saved query_{}.bin ({} bytes)", index, query_bytes.len());

        // Save expected plaintext (the raw tile bytes)
        fs::write(
            output_dir.join(format!("expected/plaintext_{}.bin", index)),
            &tiles[index],
        )?;
        info!("  Saved plaintext_{}.bin", index);
    }

    info!("");
    info!("Test vectors generated successfully!");
    info!("Output directory: {:?}", output_dir);

    Ok(())
}

/// Generate synthetic tiles with deterministic random data.
fn generate_synthetic_tiles<R: Rng>(
    num_tiles: usize,
    tile_size: usize,
    rng: &mut R,
) -> Vec<Vec<u8>> {
    (0..num_tiles)
        .map(|i| {
            let mut tile = vec![0u8; tile_size];
            // First 4 bytes: tile index (for easy identification)
            tile[..4].copy_from_slice(&(i as u32).to_le_bytes());
            // Rest: random data
            rng.fill(&mut tile[4..]);
            tile
        })
        .collect()
}

/// Create a PIR query ciphertext for a specific index.
fn create_pir_query<R: Rng>(
    index: usize,
    element_size: usize,
    params: &Arc<BfvParameters>,
    sk: &SecretKey,
    dim1: usize,
    dim2: usize,
    elements_per_plaintext: usize,
    rng: &mut R,
) -> Result<Ciphertext> {
    let plaintext_modulus = params.plaintext();
    let level = (dim1 + dim2).next_power_of_two().ilog2();

    // Compute which row this index maps to in the packed database
    let query_index = index / elements_per_plaintext;

    // Create selection vector
    let mut pt = vec![0u64; dim1 + dim2];
    let inv = inverse(1 << level, plaintext_modulus)
        .ok_or_else(|| anyhow::anyhow!("Failed to compute inverse"))?;

    // Set selection bits
    pt[query_index / dim2] = inv;        // Row selection (dim1)
    pt[dim1 + (query_index % dim2)] = inv; // Column selection (dim2)

    // Encode and encrypt
    let query_pt = Plaintext::try_encode(&pt, bfv::Encoding::poly_at_level(1), params)
        .context("Failed to encode query plaintext")?;

    let query: Ciphertext = sk.try_encrypt(&query_pt, rng)
        .context("Failed to encrypt query")?;

    Ok(query)
}
