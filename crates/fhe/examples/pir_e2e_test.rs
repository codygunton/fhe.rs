//! Headless end-to-end test for the PIR pipeline against a running MulPIR GPU
//! server.
//!
//! Generates evaluation keys (Galois + Relin) using RNS decomposition
//! (METHOD_I) in HEonGPU's native binary format, sends them over TCP, creates
//! an encrypted query, and verifies the decrypted response matches the expected
//! tile data from `test_vectors/tiles.bin`.
//!
//! Requires a running GPU server on `--host`:`--port` (default localhost:8080).
//!
//! Usage:
//!   cargo run --release -p fhe --example pir_e2e_test -- \
//!     --test-vectors-dir mulpir-gpu-server/test_vectors
#![expect(
    clippy::indexing_slicing,
    reason = "performance or example code relies on validated indices"
)]

mod util;

use clap::Parser;
use fhe::bfv::{self, BfvParametersBuilder, Ciphertext, Encoding, Plaintext, SecretKey};
use fhe_math::rq::{self, traits::TryConvertFrom, Representation};
use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter, Serialize};
use fhe_util::{inverse, transcode_to_bytes};
use prost::Message;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// ─── BFV parameter constants (must match WASM client + GPU server) ──────────

const DEGREE: usize = 8192;
const PLAINTEXT_MODULUS: u64 = (1 << 20) + (1 << 19) + (1 << 17) + (1 << 16) + (1 << 14) + 1;
const Q_MODULI: [u64; 3] = [0x3FFFFFFFFC001, 0x7FFFFFFFFB4001, 0x7FFFFFFFEAC001];
const P_MODULUS: u64 = 0x7FFFFFFFE90001;
const Q_SIZE: usize = 3;
const Q_PRIME_SIZE: usize = 4; // Q_SIZE + 1 P modulus

/// Size of one KSK entry for METHOD_I: 2 * Q_SIZE * Q_PRIME_SIZE * DEGREE.
const KSK_ENTRY_SIZE: usize = 2 * Q_SIZE * Q_PRIME_SIZE * DEGREE;

// ─── HEonGPU serialization constants ────────────────────────────────────────

const HEONGPU_SCHEME_BFV: u8 = 0x01;
const HEONGPU_KS_METHOD_I: u8 = 0x01;
const HEONGPU_STORAGE_HOST: u8 = 0x01;

// ─── Wire protocol constants ────────────────────────────────────────────────

const MSG_SET_GALOIS_KEY: u32 = 0x01;
const MSG_SET_RELIN_KEY: u32 = 0x02;
const MSG_QUERY: u32 = 0x03;

const STATUS_OK: u32 = 0x00;

#[derive(Parser)]
#[command(name = "pir_e2e_test")]
#[command(about = "Headless E2E test for PIR pipeline against MulPIR GPU server")]
struct Args {
    /// GPU server hostname.
    #[arg(long, default_value = "localhost")]
    host: String,

    /// GPU server port.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Path to test_vectors directory containing tiles.bin and params.json.
    #[arg(long, default_value = "mulpir-gpu-server/test_vectors")]
    test_vectors_dir: PathBuf,

    /// Tile indices to test (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "0")]
    tile_indices: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("PIR E2E Test");
    println!("  Server: {}:{}", args.host, args.port);
    println!("  Test vectors: {:?}", args.test_vectors_dir);
    println!("  Tile indices: {:?}", args.tile_indices);
    println!();

    // ── Load test vectors ───────────────────────────────────────────────

    let tiles_path = args.test_vectors_dir.join("tiles.bin");
    let tiles_data = std::fs::read(&tiles_path)?;
    let num_tiles = u64::from_le_bytes(tiles_data[..8].try_into()?) as usize;
    let tile_size = u64::from_le_bytes(tiles_data[8..16].try_into()?) as usize;
    println!("Database: {num_tiles} tiles x {tile_size} bytes");

    // ── Build BFV parameters ────────────────────────────────────────────

    let params = BfvParametersBuilder::new()
        .set_degree(DEGREE)
        .set_plaintext_modulus(PLAINTEXT_MODULUS)
        .set_moduli_sizes(&[50, 55, 55])
        .build_arc()?;

    let plaintext_nbits = PLAINTEXT_MODULUS.ilog2() as usize;
    let elements_per_plaintext =
        util::number_elements_per_plaintext(DEGREE, plaintext_nbits, tile_size);
    let num_rows = num_tiles.div_ceil(elements_per_plaintext);
    let dim1 = (num_rows as f64).sqrt().ceil() as usize;
    let dim2 = num_rows.div_ceil(dim1);
    let expansion_level = (dim1 + dim2).next_power_of_two().ilog2() as usize;

    println!("PIR: dim1={dim1}, dim2={dim2}, expansion_level={expansion_level}");
    println!("  elements_per_plaintext={elements_per_plaintext}, num_rows={num_rows}");
    println!();

    // ── Build Q_tilda context ───────────────────────────────────────────

    let q_tilda_moduli = [Q_MODULI[0], Q_MODULI[1], Q_MODULI[2], P_MODULUS];
    let ctx_q_tilda = rq::Context::new_arc(&q_tilda_moduli, DEGREE)?;

    let ctx_level0 = params.context_at_level(0)?.clone();

    // ── Precompute HEonGPU NTT tables ───────────────────────────────────

    let ntt_conv = NttConverter::new(&q_tilda_moduli, DEGREE);
    println!(
        "HEonGPU NTT roots (psi): {:?}",
        ntt_conv.psi_roots
    );

    // ── Generate secret key ─────────────────────────────────────────────

    let mut rng = rand::rng();
    let sk = SecretKey::random(&params, &mut rng);

    let sk_bytes = sk.to_bytes();
    let sk_proto = fhe::proto::bfv::SecretKey::decode(&sk_bytes[..])?;
    let sk_coeffs: &[i64] = &sk_proto.coeffs;

    let sk_ntt = {
        let mut poly =
            rq::Poly::try_convert_from(sk_coeffs, &ctx_q_tilda, false, Representation::PowerBasis)?;
        poly.change_representation(Representation::Ntt);
        poly
    };

    println!("Secret key generated ({} coefficients)", sk_coeffs.len());

    // ── Generate Galois key ─────────────────────────────────────────────

    let start = Instant::now();
    let galois_elts: Vec<usize> = (0..expansion_level).map(|l| (DEGREE >> l) + 1).collect();
    let factors = compute_rns_factors();

    let mut galois_keys: Vec<(usize, Vec<u64>)> = Vec::with_capacity(galois_elts.len());

    for &g in &galois_elts {
        // HEonGPU uses the INVERSE Galois element: s(X^{g^{-1}}) not s(X^g)
        let g_inv = inverse(g as u64, 2 * DEGREE as u64)
            .ok_or_else(|| format!("No inverse for g={g} mod {}", 2 * DEGREE))?
            as usize;
        let sub_exp = rq::SubstitutionExponent::new(&ctx_q_tilda, g_inv)?;
        let s_g = sk_ntt.substitute(&sub_exp)?;

        let mut key_data = vec![0u64; KSK_ENTRY_SIZE];

        for i in 0..Q_SIZE {
            let a = rq::Poly::random(&ctx_q_tilda, Representation::Ntt, &mut rng);
            let e = rq::Poly::small(&ctx_q_tilda, Representation::Ntt, 1, &mut rng)?;
            let factor_poly = make_factor_poly(i, factors[i], &ctx_q_tilda)?;
            let message = &factor_poly * &sk_ntt;
            let c0 = &(&(-&(&(&s_g * &a) + &e)) + &message);

            // Convert from fhe.rs NTT → coefficient form → HEonGPU NTT
            let c0_converted = ntt_conv.convert_poly_fhe_to_heongpu(c0);
            let c1_converted = ntt_conv.convert_poly_fhe_to_heongpu(&a);

            let base = i * 2 * Q_PRIME_SIZE * DEGREE;
            key_data[base..base + Q_PRIME_SIZE * DEGREE]
                .copy_from_slice(&c0_converted);
            key_data[base + Q_PRIME_SIZE * DEGREE..base + 2 * Q_PRIME_SIZE * DEGREE]
                .copy_from_slice(&c1_converted);
        }

        galois_keys.push((g, key_data));
    }

    let galois_key_bytes = serialize_galois_key_heongpu(&galois_elts, &galois_keys);
    println!(
        "Galois key generated: {} bytes ({:.1}s)",
        galois_key_bytes.len(),
        start.elapsed().as_secs_f64()
    );

    // ── Generate Relin key ──────────────────────────────────────────────

    let start = Instant::now();
    let s_sq = &sk_ntt * &sk_ntt;
    let mut relin_data = vec![0u64; KSK_ENTRY_SIZE];

    for i in 0..Q_SIZE {
        let a = rq::Poly::random(&ctx_q_tilda, Representation::Ntt, &mut rng);
        let e = rq::Poly::small(&ctx_q_tilda, Representation::Ntt, 1, &mut rng)?;
        let factor_poly = make_factor_poly(i, factors[i], &ctx_q_tilda)?;
        let message = &factor_poly * &s_sq;
        let c0 = &(&(-&(&(&sk_ntt * &a) + &e)) + &message);

        let c0_converted = ntt_conv.convert_poly_fhe_to_heongpu(c0);
        let c1_converted = ntt_conv.convert_poly_fhe_to_heongpu(&a);

        let base = i * 2 * Q_PRIME_SIZE * DEGREE;
        relin_data[base..base + Q_PRIME_SIZE * DEGREE]
            .copy_from_slice(&c0_converted);
        relin_data[base + Q_PRIME_SIZE * DEGREE..base + 2 * Q_PRIME_SIZE * DEGREE]
            .copy_from_slice(&c1_converted);
    }

    let relin_key_bytes = serialize_relin_key_heongpu(&relin_data);
    println!(
        "Relin key generated: {} bytes ({:.1}s)",
        relin_key_bytes.len(),
        start.elapsed().as_secs_f64()
    );
    println!();

    // ── Send keys to GPU server ─────────────────────────────────────────

    let addr = format!("{}:{}", args.host, args.port);

    print!("Sending Galois key... ");
    let start = Instant::now();
    let (status, _) = send_to_gpu(&addr, MSG_SET_GALOIS_KEY, &galois_key_bytes)?;
    if status != STATUS_OK {
        return Err(format!("Galois key rejected: status=0x{status:02x}").into());
    }
    println!("OK ({:.1}s)", start.elapsed().as_secs_f64());

    print!("Sending Relin key... ");
    let start = Instant::now();
    let (status, _) = send_to_gpu(&addr, MSG_SET_RELIN_KEY, &relin_key_bytes)?;
    if status != STATUS_OK {
        return Err(format!("Relin key rejected: status=0x{status:02x}").into());
    }
    println!("OK ({:.1}s)", start.elapsed().as_secs_f64());
    println!();

    // ── Test queries ────────────────────────────────────────────────────

    let mut pass_count = 0;
    let mut fail_count = 0;

    for &tile_index in &args.tile_indices {
        if tile_index >= num_tiles {
            println!("SKIP tile {tile_index} (out of range, max={num_tiles})");
            continue;
        }

        print!("Query tile {tile_index}... ");
        let start = Instant::now();

        // Create query
        let query_index = tile_index / elements_per_plaintext;
        let inv = inverse(1u64 << expansion_level, PLAINTEXT_MODULUS)
            .ok_or("No modular inverse")?;

        let mut pt = vec![0u64; dim1 + dim2];
        pt[query_index / dim2] = inv;
        pt[dim1 + (query_index % dim2)] = inv;

        let query_pt = Plaintext::try_encode(&pt, Encoding::poly(), &params)?;
        let query_ct: Ciphertext = sk.try_encrypt(&query_pt, &mut rng)?;
        let query_bytes = serialize_ct_heongpu(&query_ct);

        // Send query
        let (status, response_bytes) = send_to_gpu(&addr, MSG_QUERY, &query_bytes)?;
        if status != STATUS_OK {
            println!("FAIL (query rejected: status=0x{status:02x})");
            fail_count += 1;
            continue;
        }

        // Decrypt response — first print diagnostics
        {
            let resp_ring = u32::from_le_bytes(response_bytes[1..5].try_into().unwrap());
            let resp_mods = u32::from_le_bytes(response_bytes[5..9].try_into().unwrap());
            let resp_ct_sz = u32::from_le_bytes(response_bytes[9..13].try_into().unwrap());
            let resp_ntt = response_bytes[13];
            let resp_total = u32::from_le_bytes(response_bytes[17..21].try_into().unwrap());
            eprintln!(
                "  Response header: ring={resp_ring}, mods={resp_mods}, ct_size={resp_ct_sz}, in_ntt={resp_ntt}, total={resp_total}, bytes={}",
                response_bytes.len()
            );
        }

        let response = deserialize_ct_heongpu(&response_bytes, &params, &ctx_level0)?;
        let pt_result = sk.try_decrypt(&response)?;
        let coeffs = Vec::<u64>::try_decode(&pt_result, Encoding::poly())?;

        // Print first 8 decrypted polynomial coefficients
        eprintln!(
            "  Decrypted coeffs[0..8]: {:?}",
            &coeffs[..8.min(coeffs.len())]
        );

        let bytes = transcode_to_bytes(&coeffs, 20);

        let offset = tile_index % elements_per_plaintext;
        let tile_start = offset * tile_size;
        let tile_end = tile_start + tile_size;

        if tile_end > bytes.len() {
            println!("FAIL (tile offset out of range)");
            fail_count += 1;
            continue;
        }

        let decrypted_tile = &bytes[tile_start..tile_end];

        // Compare against expected tile data from tiles.bin
        let expected_start = 16 + tile_index * tile_size;
        let expected_end = expected_start + tile_size;
        let expected_tile = &tiles_data[expected_start..expected_end];

        if decrypted_tile == expected_tile {
            println!(
                "PASS ({:.1}s, first 4 bytes: {:02x}{:02x}{:02x}{:02x})",
                start.elapsed().as_secs_f64(),
                decrypted_tile[0],
                decrypted_tile[1],
                decrypted_tile[2],
                decrypted_tile[3],
            );
            pass_count += 1;
        } else {
            let diff_count = decrypted_tile
                .iter()
                .zip(expected_tile.iter())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "FAIL ({diff_count}/{tile_size} bytes differ, {:.1}s)",
                start.elapsed().as_secs_f64(),
            );
            println!(
                "  Expected first 8: {:02x?}",
                &expected_tile[..8.min(tile_size)]
            );
            println!(
                "  Got first 8:      {:02x?}",
                &decrypted_tile[..8.min(tile_size)]
            );
            fail_count += 1;
        }
    }

    // ── Summary ─────────────────────────────────────────────────────────

    println!();
    println!("Results: {pass_count} passed, {fail_count} failed");

    if fail_count > 0 {
        Err(format!("{fail_count} test(s) failed").into())
    } else {
        println!("All tests passed!");
        Ok(())
    }
}

// ─── RNS key generation helpers ─────────────────────────────────────────────

fn compute_rns_factors() -> [u64; Q_SIZE] {
    [
        P_MODULUS % Q_MODULI[0],
        P_MODULUS % Q_MODULI[1],
        P_MODULUS % Q_MODULI[2],
    ]
}

fn make_factor_poly(
    decomp_idx: usize,
    factor_value: u64,
    ctx: &Arc<rq::Context>,
) -> Result<rq::Poly, Box<dyn std::error::Error>> {
    let mut data = vec![0u64; Q_PRIME_SIZE * DEGREE];
    for k in 0..DEGREE {
        data[decomp_idx * DEGREE + k] = factor_value;
    }
    Ok(rq::Poly::try_convert_from(data, ctx, true, Representation::Ntt)?)
}

// ─── HEonGPU NTT conversion ────────────────────────────────────────────────
//
// fhe.rs and HEonGPU use different NTT primitive roots. HEonGPU uses the
// *minimal* primitive 2N-th root of unity for each modulus, while fhe.rs
// uses a deterministic but non-minimal root. Key data must be converted:
//   fhe.rs NTT → INTT(fhe.rs) → coefficient form → NTT(HEonGPU)

/// Modular multiplication using 128-bit intermediate.
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation via square-and-multiply.
fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    result
}

/// Check if `root` is a primitive `degree`-th root of unity mod `m`.
/// For degree = 2N: root^N ≡ m-1 (mod m).
fn is_primitive_root(root: u64, degree: usize, m: u64) -> bool {
    if root == 0 {
        return false;
    }
    mod_pow(root, (degree / 2) as u64, m) == m - 1
}

/// Find the minimal primitive `degree`-th root of unity mod `m`.
/// Matches HEonGPU's `find_minimal_primitive_root()`.
fn find_minimal_primitive_root(degree: usize, m: u64) -> u64 {
    let quotient = (m - 1) / degree as u64;

    // Find any primitive root by trying small candidates
    let mut root = 0u64;
    for candidate in 2..m {
        let r = mod_pow(candidate, quotient, m);
        if is_primitive_root(r, degree, m) {
            root = r;
            break;
        }
    }
    assert!(root != 0, "No primitive root found for m={m}, degree={degree}");

    // Find minimum among all primitive roots (odd powers of root).
    // Primitive roots of order `degree` (a power of 2) are exactly
    // the odd powers: root^1, root^3, root^5, ...
    let gen_sq = mod_mul(root, root, m);
    let mut current = root;
    let mut min_root = root;
    // Iterate degree/2 times (stepping by 2 through 0..degree)
    for _ in (0..degree).step_by(2) {
        if current < min_root {
            min_root = current;
        }
        current = mod_mul(current, gen_sq, m);
    }
    min_root
}

fn bit_reverse(x: usize, bits: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - bits)
}

/// Precomputed NTT tables for converting between fhe.rs and HEonGPU NTT forms.
struct NttConverter {
    /// Forward NTT tables per modulus: psi powers in bit-reversed order.
    forward_tables: Vec<Vec<u64>>,
    /// The moduli (Q + P).
    moduli: Vec<u64>,
    /// Degree N.
    n: usize,
    /// Minimal primitive roots (for debugging).
    psi_roots: Vec<u64>,
}

impl NttConverter {
    fn new(moduli: &[u64], n: usize) -> Self {
        let n_power = n.ilog2();
        let degree_2n = 2 * n;
        let mut forward_tables = Vec::with_capacity(moduli.len());
        let mut psi_roots = Vec::with_capacity(moduli.len());

        for &m in moduli {
            let psi = find_minimal_primitive_root(degree_2n, m);
            psi_roots.push(psi);

            // Build powers: [1, psi, psi^2, ..., psi^(N-1)]
            let mut powers = Vec::with_capacity(n);
            let mut p = 1u64;
            for _ in 0..n {
                powers.push(p);
                p = mod_mul(p, psi, m);
            }

            // Bit-reverse the table
            let mut table = vec![0u64; n];
            for j in 0..n {
                table[j] = powers[bit_reverse(j, n_power)];
            }
            forward_tables.push(table);
        }

        Self {
            forward_tables,
            moduli: moduli.to_vec(),
            n,
            psi_roots,
        }
    }

    /// Apply in-place negacyclic NTT (Cooley-Tukey) to a single modulus slice.
    fn ntt_inplace(&self, data: &mut [u64], modulus_idx: usize) {
        let n = self.n;
        let m = self.moduli[modulus_idx];
        let table = &self.forward_tables[modulus_idx];

        let mut t = n;
        let mut mval = 1;
        let n_power = n.ilog2();

        for _ in 0..n_power {
            t >>= 1;
            for i in 0..mval {
                let j1 = 2 * i * t;
                let j2 = j1 + t;
                let s = table[mval + i];
                for j in j1..j2 {
                    let u = data[j];
                    let v = mod_mul(data[j + t], s, m);
                    data[j] = if u + v >= m { u + v - m } else { u + v };
                    data[j + t] = if u >= v { u - v } else { u + m - v };
                }
            }
            mval <<= 1;
        }
    }

    /// Convert a fhe.rs NTT-form polynomial to HEonGPU NTT form.
    ///
    /// 1. Clone poly and INTT (fhe.rs) → coefficient form
    /// 2. Apply HEonGPU forward NTT per modulus
    fn convert_poly_fhe_to_heongpu(&self, poly: &rq::Poly) -> Vec<u64> {
        // Step 1: Convert to coefficient form
        let mut coeff_poly = poly.clone();
        coeff_poly.change_representation(Representation::PowerBasis);
        let raw = coeff_poly.coefficients();
        let slice = raw.as_slice().unwrap();

        // Step 2: Apply HEonGPU NTT per modulus
        let num_moduli = self.moduli.len();
        let mut result = slice.to_vec();
        for i in 0..num_moduli {
            let start = i * self.n;
            let end = start + self.n;
            self.ntt_inplace(&mut result[start..end], i);
        }
        result
    }
}

// ─── HEonGPU serialization ─────────────────────────────────────────────────

fn serialize_galois_key_heongpu(
    galois_elts: &[usize],
    keys: &[(usize, Vec<u64>)],
) -> Vec<u8> {
    let galois_elt_zero: i32 = (2 * DEGREE - 1) as i32;
    let galoiskey_size: u64 = KSK_ENTRY_SIZE as u64;

    let data_per_key = 4 + KSK_ENTRY_SIZE * 8;
    let capacity = 64 + galois_elts.len() * 4 + keys.len() * data_per_key + KSK_ENTRY_SIZE * 8;
    let mut buf = Vec::with_capacity(capacity);

    buf.push(HEONGPU_SCHEME_BFV);
    buf.push(HEONGPU_KS_METHOD_I);
    buf.extend_from_slice(&(DEGREE as i32).to_le_bytes());
    buf.extend_from_slice(&(Q_PRIME_SIZE as i32).to_le_bytes());
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes());
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes()); // d_
    buf.push(0x01); // customized = true
    buf.extend_from_slice(&3i32.to_le_bytes()); // group_order_
    buf.push(HEONGPU_STORAGE_HOST);
    buf.push(0x01); // galois_key_generated_

    buf.extend_from_slice(&(galois_elts.len() as u32).to_le_bytes());
    for &g in galois_elts {
        buf.extend_from_slice(&(g as u32).to_le_bytes());
    }

    buf.extend_from_slice(&galois_elt_zero.to_le_bytes());
    buf.extend_from_slice(&galoiskey_size.to_le_bytes());

    buf.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for &(g, ref data) in keys {
        buf.extend_from_slice(&(g as i32).to_le_bytes());
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    // Zero key data (column rotation — unused for expansion)
    buf.extend_from_slice(&vec![0u8; KSK_ENTRY_SIZE * 8]);

    buf
}

fn serialize_relin_key_heongpu(key_data: &[u64]) -> Vec<u8> {
    let relinkey_size: u64 = KSK_ENTRY_SIZE as u64;

    let mut buf = Vec::with_capacity(36 + KSK_ENTRY_SIZE * 8);

    buf.push(HEONGPU_SCHEME_BFV);
    buf.push(HEONGPU_KS_METHOD_I);
    buf.extend_from_slice(&(DEGREE as i32).to_le_bytes());
    buf.extend_from_slice(&(Q_PRIME_SIZE as i32).to_le_bytes());
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes());
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes()); // d_
    buf.extend_from_slice(&(Q_PRIME_SIZE as i32).to_le_bytes()); // d_tilda_
    buf.extend_from_slice(&1i32.to_le_bytes()); // r_prime_
    buf.push(HEONGPU_STORAGE_HOST);
    buf.push(0x01); // relin_key_generated_
    buf.extend_from_slice(&relinkey_size.to_le_bytes());

    for &v in key_data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

fn serialize_ct_heongpu(ct: &Ciphertext) -> Vec<u8> {
    let cipher_size = ct.len() as u32;
    let num_moduli = ct[0].coefficients().nrows() as u32;
    let ring_size = ct[0].coefficients().ncols() as u32;
    let total_coeffs = cipher_size * num_moduli * ring_size;

    let mut buf = Vec::with_capacity(21 + (total_coeffs as usize) * 8);

    buf.push(HEONGPU_SCHEME_BFV);
    buf.extend_from_slice(&ring_size.to_le_bytes());
    buf.extend_from_slice(&num_moduli.to_le_bytes());
    buf.extend_from_slice(&cipher_size.to_le_bytes());
    buf.push(0x00); // in_ntt_domain_ = false (BFV uses coefficient form)
    buf.push(HEONGPU_STORAGE_HOST); // storage_type_
    buf.push(0x00); // relinearization_required_
    buf.push(0x01); // ciphertext_generated_
    buf.extend_from_slice(&total_coeffs.to_le_bytes());

    // HEonGPU BFV expects coefficient form; fhe.rs stores NTT internally.
    for poly in ct.iter() {
        let mut coeff_poly = poly.clone();
        coeff_poly.change_representation(Representation::PowerBasis);
        for &c in coeff_poly.coefficients().as_slice().unwrap() {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }

    buf
}

fn deserialize_ct_heongpu(
    bytes: &[u8],
    params: &Arc<bfv::BfvParameters>,
    ctx: &Arc<rq::Context>,
) -> Result<Ciphertext, Box<dyn std::error::Error>> {
    if bytes.len() < 21 {
        return Err("Ciphertext too short for header".into());
    }

    let scheme = bytes[0];
    if scheme != HEONGPU_SCHEME_BFV {
        return Err(format!("Expected BFV scheme (0x01), got 0x{scheme:02x}").into());
    }

    let ring_size = u32::from_le_bytes(bytes[1..5].try_into()?) as usize;
    let num_moduli = u32::from_le_bytes(bytes[5..9].try_into()?) as usize;
    let cipher_size = u32::from_le_bytes(bytes[9..13].try_into()?) as usize;
    let in_ntt = bytes[13] != 0;
    let total_coeffs = u32::from_le_bytes(bytes[17..21].try_into()?) as usize;

    let expected_total = cipher_size * num_moduli * ring_size;
    if total_coeffs != expected_total {
        return Err(format!(
            "Ciphertext size mismatch: header says {total_coeffs}, expected {expected_total}"
        )
        .into());
    }

    let data_bytes = 21 + total_coeffs * 8;
    if bytes.len() < data_bytes {
        return Err(format!(
            "Ciphertext data too short: need {data_bytes}, have {}",
            bytes.len()
        )
        .into());
    }

    let coeffs_per_poly = num_moduli * ring_size;
    let repr = if in_ntt {
        Representation::Ntt
    } else {
        Representation::PowerBasis
    };

    // HEonGPU keeps cipher_size=3 after relinearization (stale c2 remains).
    // Its decrypt() only uses c0+c1*s, so we only take the first 2 components.
    let use_size = cipher_size.min(2);

    let mut polys = Vec::with_capacity(use_size);
    let mut offset = 21;

    for i in 0..cipher_size {
        let mut raw = vec![0u64; coeffs_per_poly];
        for v in raw.iter_mut() {
            *v = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
            offset += 8;
        }

        if i < use_size {
            let mut poly = rq::Poly::try_convert_from(raw, ctx, false, repr)?;
            if !in_ntt {
                poly.change_representation(Representation::Ntt);
            }
            polys.push(poly);
        }
    }

    Ok(Ciphertext::new(polys, params)?)
}

// ─── TCP wire protocol ──────────────────────────────────────────────────────

fn send_to_gpu(
    addr: &str,
    msg_type: u32,
    payload: &[u8],
) -> Result<(u32, Vec<u8>), Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;

    let mut header = [0u8; 8];
    header[..4].copy_from_slice(&msg_type.to_le_bytes());
    header[4..8].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(payload)?;

    let mut resp_header = [0u8; 8];
    stream.read_exact(&mut resp_header)?;
    let status = u32::from_le_bytes(resp_header[..4].try_into()?);
    let resp_len = u32::from_le_bytes(resp_header[4..8].try_into()?) as usize;

    let mut resp = vec![0u8; resp_len];
    if resp_len > 0 {
        stream.read_exact(&mut resp)?;
    }

    Ok((status, resp))
}
