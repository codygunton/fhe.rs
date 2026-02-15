//! Generates test vectors for MulPIR GPU server validation.
//!
//! This utility creates a deterministic set of test data including:
//! - Synthetic database tiles (tiles.bin)
//! - Secret key raw coefficients (secret_key.bin)
//! - PIR parameters as JSON (params.json)
//! - Expected plaintext results (expected_{idx}.bin)
//!
//! The C++ side (HEonGPU) loads these to verify the GPU PIR implementation
//! produces correct results by importing the secret key, generating its own
//! query, and comparing decrypted output against expected results.
//!
//! Run with:
//!   cargo run --release -p fhe --example generate_test_vectors -- \
//!     --output-dir mulpir-gpu-server/test_vectors --num-tiles 100 --tile-size 20480
#![expect(
    clippy::indexing_slicing,
    reason = "performance or example code relies on validated indices"
)]

mod util;

use clap::Parser;
use fhe::bfv::{self, BfvParametersBuilder};
use fhe_traits::Serialize;
use prost::Message;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
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

    /// Size of each tile in bytes (max 20480 for degree=8192, t=1785857).
    #[arg(long, default_value = "20480")]
    tile_size: usize,

    /// Random seed for deterministic generation.
    #[arg(long, default_value = "12345")]
    seed: u64,

    /// Indices to generate expected results for (comma-separated).
    /// If not specified, uses: 0, 1, dim1, dim1*dim2-1, 42
    #[arg(long, value_delimiter = ',')]
    query_indices: Option<Vec<usize>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("MulPIR Test Vector Generator");
    println!("  Output: {:?}", args.output_dir);
    println!(
        "  Tiles: {} x {} bytes",
        args.num_tiles, args.tile_size
    );
    println!("  Seed: {}", args.seed);

    // BFV parameters matching the GPU server
    let degree = 8192;
    let plaintext_modulus: u64 = (1 << 20) + (1 << 19) + (1 << 17) + (1 << 16) + (1 << 14) + 1;
    let moduli_sizes = [50, 55, 55];

    // Validate parameters
    let max_element_size = ((plaintext_modulus.ilog2() as usize) * degree) / 8;
    if args.tile_size > max_element_size {
        return Err(format!(
            "Tile size {} exceeds maximum {} for these BFV parameters",
            args.tile_size, max_element_size
        )
        .into());
    }

    // Initialize deterministic RNG
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);

    // Generate synthetic tiles (deterministic based on seed)
    println!("Generating {} synthetic tiles...", args.num_tiles);
    let tiles = generate_synthetic_tiles(args.num_tiles, args.tile_size, &mut rng);

    // Build BFV parameters
    println!("Building BFV parameters...");
    let params = BfvParametersBuilder::new()
        .set_degree(degree)
        .set_plaintext_modulus(plaintext_modulus)
        .set_moduli_sizes(&moduli_sizes)
        .build_arc()?;

    // Extract actual Q prime values from the built parameters
    let q_primes: Vec<u64> = params.moduli().to_vec();
    println!("Q primes: {:?}", q_primes);
    for (i, q) in q_primes.iter().enumerate() {
        println!("  q[{}] = {} (0x{:X})", i, q, q);
    }

    // Compute PIR dimensions
    let plaintext_nbits = plaintext_modulus.ilog2() as usize;
    let elements_per_plaintext =
        number_elements_per_plaintext(degree, plaintext_nbits, args.tile_size);
    let num_rows = args.num_tiles.div_ceil(elements_per_plaintext);
    let dim1 = (num_rows as f64).sqrt().ceil() as usize;
    let dim2 = num_rows.div_ceil(dim1);
    let expansion_level = (dim1 + dim2).next_power_of_two().ilog2() as usize;

    println!(
        "PIR dimensions: {}x{} (expansion level {})",
        dim1, dim2, expansion_level
    );
    println!("  elements_per_plaintext = {}", elements_per_plaintext);
    println!("  num_rows = {}", num_rows);

    // Generate secret key
    println!("Generating secret key...");
    let sk = bfv::SecretKey::random(&params, &mut rng);

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

    println!("Query indices: {:?}", query_indices);

    // Create output directory
    let output_dir = &args.output_dir;
    fs::create_dir_all(output_dir)?;

    // --- Export secret_key.bin ---
    // SecretKey.coeffs is pub(crate), so we serialize to protobuf and decode
    // to extract raw i64 coefficients.
    println!("Exporting secret key...");
    let sk_bytes = sk.to_bytes();
    let sk_proto = fhe::proto::bfv::SecretKey::decode(&sk_bytes[..])?;
    let sk_coeffs: &[i64] = &sk_proto.coeffs;
    println!(
        "  Secret key: {} coefficients (expected {})",
        sk_coeffs.len(),
        degree
    );

    // Display coefficient distribution (CBD sampling, centered around 0)
    let (neg_count, zero_count, pos_count) = sk_coeffs.iter().fold((0usize, 0usize, 0usize), |(n, z, p), &c| {
        match c {
            -1 => (n + 1, z, p),
            0 => (n, z + 1, p),
            1 => (n, z, p + 1),
            _ => {
                // CBD distribution can produce values beyond -1,0,1
                (n, z, p)
            }
        }
    });
    println!(
        "  Coefficient distribution: -1:{}, 0:{}, +1:{}, other:{}",
        neg_count,
        zero_count,
        pos_count,
        sk_coeffs.len() - neg_count - zero_count - pos_count
    );

    // Write secret_key.bin: [num_coeffs: u64][coeff_0: i64]...[coeff_N-1: i64]
    {
        let sk_path = output_dir.join("secret_key.bin");
        let mut f = File::create(&sk_path)?;
        f.write_all(&(sk_coeffs.len() as u64).to_le_bytes())?;
        for &coeff in sk_coeffs {
            f.write_all(&coeff.to_le_bytes())?;
        }
        let file_size = (1 + sk_coeffs.len()) * 8;
        println!("  Saved secret_key.bin ({} bytes)", file_size);
    }

    // --- Export tiles.bin ---
    // Format: [num_tiles: u64][tile_size: u64][tile_0_bytes...][tile_1_bytes...]
    println!("Exporting tiles...");
    {
        let tiles_path = output_dir.join("tiles.bin");
        let mut f = File::create(&tiles_path)?;
        f.write_all(&(args.num_tiles as u64).to_le_bytes())?;
        f.write_all(&(args.tile_size as u64).to_le_bytes())?;
        for tile in &tiles {
            f.write_all(tile)?;
        }
        let file_size = 16 + tiles.len() * args.tile_size;
        println!("  Saved tiles.bin ({} bytes)", file_size);
    }

    // --- Export params.json ---
    // Write JSON manually to avoid serde/serde_json dependency.
    println!("Exporting parameters...");
    {
        let q_primes_json: Vec<String> = q_primes.iter().map(|q| q.to_string()).collect();
        let query_indices_json: Vec<String> =
            query_indices.iter().map(|i| i.to_string()).collect();

        let json = format!(
            r#"{{
  "poly_degree": {},
  "plaintext_modulus": {},
  "q_primes": [{}],
  "num_tiles": {},
  "tile_size": {},
  "elements_per_plaintext": {},
  "num_rows": {},
  "dim1": {},
  "dim2": {},
  "expansion_level": {},
  "query_indices": [{}]
}}"#,
            degree,
            plaintext_modulus,
            q_primes_json.join(", "),
            args.num_tiles,
            args.tile_size,
            elements_per_plaintext,
            num_rows,
            dim1,
            dim2,
            expansion_level,
            query_indices_json.join(", "),
        );

        let params_path = output_dir.join("params.json");
        fs::write(&params_path, &json)?;
        println!("  Saved params.json");
    }

    // --- Export expected_{idx}.bin ---
    // Each file contains the raw tile bytes for verification after PIR retrieval.
    // The C++ side decrypts the PIR response, extracts the correct tile from the
    // packed plaintext row at the given offset, and compares against these bytes.
    println!("Exporting expected results...");
    for &index in &query_indices {
        if index >= args.num_tiles {
            println!("  WARNING: Skipping index {} (out of range)", index);
            continue;
        }

        // The PIR server packs multiple tiles per plaintext row. The C++ side
        // needs to know the offset within the decoded row to extract this tile.
        let offset_in_row = index % elements_per_plaintext;

        // Format: [offset_in_row: u64][raw_tile_bytes...]
        let expected_path = output_dir.join(format!("expected_{}.bin", index));
        let mut f = File::create(&expected_path)?;
        f.write_all(&(offset_in_row as u64).to_le_bytes())?;
        f.write_all(&tiles[index])?;
        println!(
            "  Saved expected_{}.bin ({} bytes, offset_in_row={})",
            index,
            8 + args.tile_size,
            offset_in_row
        );
    }

    println!();
    println!("Test vectors generated successfully!");
    println!("Output directory: {:?}", output_dir);
    println!();
    println!("Files:");
    println!("  params.json       - PIR dimensions, Q primes, query indices");
    println!("  secret_key.bin    - Raw i64 coefficients for SK import");
    println!("  tiles.bin         - Database tiles with header");
    for &index in &query_indices {
        if index < args.num_tiles {
            println!(
                "  expected_{}.bin  - Expected tile bytes for index {}",
                index, index
            );
        }
    }

    Ok(())
}

/// Generate synthetic tiles with deterministic random data.
///
/// Each tile starts with a 4-byte little-endian index for easy identification,
/// followed by random bytes.
fn generate_synthetic_tiles(
    num_tiles: usize,
    tile_size: usize,
    rng: &mut impl RngCore,
) -> Vec<Vec<u8>> {
    (0..num_tiles)
        .map(|i| {
            let mut tile = vec![0u8; tile_size];
            // First 4 bytes: tile index (for easy identification)
            tile[..4].copy_from_slice(&(i as u32).to_le_bytes());
            // Rest: random data
            rng.fill_bytes(&mut tile[4..]);
            tile
        })
        .collect()
}
