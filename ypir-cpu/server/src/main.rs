use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use actix_web::{web, App, HttpResponse, HttpServer};
use anyhow::{Context, Result};
use clap::Parser;
use log::{info, warn};
use rayon::prelude::*;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::params::Params;
use spiral_rs::poly::{PolyMatrix, PolyMatrixNTT};

use ypir::params::{params_for_scenario_simplepir, GetQPrime};
use ypir::server::{DbRowsPadded, OfflinePrecomputedValues, YServer};

#[derive(Parser)]
#[command(name = "ypir-cpu-server")]
struct Cli {
    /// Path to tiles.bin database file
    #[arg(long)]
    database: String,

    /// Path to tile_mapping.json
    #[arg(long)]
    tile_mapping: String,

    /// Number of PIR slots (tiles) in the database
    #[arg(long)]
    num_tiles: usize,

    /// Size of each tile in bytes
    #[arg(long, default_value_t = 20480)]
    tile_size: usize,

    /// Port to listen on
    #[arg(long, default_value_t = 8084)]
    port: u16,
}

struct ServerState {
    params: &'static Params,
    y_server: &'static YServer<'static, u16>,
    offline_vals: OfflinePrecomputedValues<'static>,
    sessions: Mutex<HashMap<String, SessionData>>,
    // Pre-computed sizes
    setup_bytes: usize,
    query_bytes: usize,
    response_bytes: usize,
    num_items: usize,
    instances: usize,
    tile_size: usize,
    ypir_params_json: String,
    rlwe_q_prime_1: u64,
    rlwe_q_prime_2: u64,
}

struct SessionData {
    /// Condensed row-1 expansion key params
    pack_pub_params_row_1s: Vec<PolyMatrixNTT<'static>>,
}

// SAFETY: YServer and OfflinePrecomputedValues use &'static Params.
// Access to mutable state (sessions) is protected by Mutex.
unsafe impl Send for ServerState {}
unsafe impl Sync for ServerState {}

fn load_db_as_u16_iter(
    db_path: &str,
    params: &Params,
    num_tiles: usize,
    tile_size: usize,
) -> Vec<u16> {
    let db_rows = 1usize << (params.db_dim_1 + params.poly_len_log2);
    let db_cols = params.instances * params.poly_len;
    let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
    let pt_modulus = params.pt_modulus;

    info!(
        "Loading DB: {} tiles, {} bytes each, db_rows={}, db_cols={}, pt_bits={}",
        num_tiles, tile_size, db_rows, db_cols, pt_bits
    );

    // Read tile data from file
    let mut file = std::fs::File::open(db_path).expect("Failed to open database file");

    // Check for 16-byte header (num_tiles u64 LE + tile_size u64 LE)
    let mut header = [0u8; 16];
    file.read_exact(&mut header).expect("Failed to read file header");
    let file_num_tiles = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let file_tile_size = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;

    let has_header = file_num_tiles == num_tiles && file_tile_size == tile_size;
    let raw_data = if has_header {
        info!("Detected 16-byte header (num_tiles={}, tile_size={})", file_num_tiles, file_tile_size);
        let mut data = Vec::new();
        file.read_to_end(&mut data).expect("Failed to read database");
        data
    } else {
        info!("No header detected, reading from start of file");
        let mut data = header.to_vec();
        file.read_to_end(&mut data).expect("Failed to read database");
        data
    };

    info!("Read {} bytes of tile data", raw_data.len());

    // Each tile is tile_size bytes. We need to pack these into the YPIR DB format.
    // The DB is db_rows x db_cols u16 values, each mod pt_modulus.
    // Each row corresponds to one item (tile).
    // Each tile's bytes are packed into pt_bits-bit coefficients across `instances` RLWE plaintexts.
    // Total coefficients per tile = instances * poly_len
    // Bytes per tile decoded from coefficients = instances * poly_len * pt_bits / 8

    let coeffs_per_tile = db_cols; // instances * poly_len
    let bytes_from_coeffs = coeffs_per_tile * pt_bits / 8;
    info!(
        "Coefficients per tile: {}, bytes decodable: {} (tile_size: {})",
        coeffs_per_tile, bytes_from_coeffs, tile_size
    );

    let mut db = vec![0u16; db_rows * db_cols];

    for tile_idx in 0..num_tiles.min(db_rows) {
        let tile_start = tile_idx * tile_size;
        let tile_end = (tile_start + tile_size).min(raw_data.len());
        let tile_data = if tile_start < raw_data.len() {
            &raw_data[tile_start..tile_end]
        } else {
            &[]
        };

        // Pack tile bytes into pt_bits-bit coefficients
        let mut bit_offs = 0usize;
        for coeff_idx in 0..coeffs_per_tile {
            let mut val = 0u64;
            for b in 0..pt_bits {
                let byte_idx = (bit_offs + b) / 8;
                let bit_idx = (bit_offs + b) % 8;
                if byte_idx < tile_data.len() {
                    val |= (((tile_data[byte_idx] >> bit_idx) & 1) as u64) << b;
                }
            }
            val %= pt_modulus;
            db[tile_idx * db_cols + coeff_idx] = val as u16;
            bit_offs += pt_bits;
        }
    }

    info!("DB loaded: {} x {} = {} u16 values", db_rows, db_cols, db.len());
    db
}

fn deserialize_pub_params(
    params: &'static Params,
    data: &[u8],
) -> Vec<PolyMatrixNTT<'static>> {
    let num_params = params.poly_len_log2;
    let cols_per = params.t_exp_left;
    let elems_per_param = cols_per * params.poly_len; // u64s per condensed matrix
    let bytes_per_param = elems_per_param * 8;

    let mut result = Vec::new();
    for i in 0..num_params {
        let start = i * bytes_per_param;
        let end = start + bytes_per_param;
        if end > data.len() {
            warn!("Pub params truncated at param {}: need {} bytes, have {}", i, end, data.len());
            break;
        }
        let chunk = &data[start..end];

        let mut mat = PolyMatrixNTT::zero(params, 1, cols_per);
        for j in 0..cols_per {
            let poly = mat.get_poly_mut(0, j);
            let poly_start = j * params.poly_len * 8;
            for z in 0..params.poly_len {
                let byte_start = poly_start + z * 8;
                let val = u64::from_le_bytes(
                    chunk[byte_start..byte_start + 8].try_into().unwrap()
                );
                // Condensed format: CRT0 in lower 32 bits, CRT1 in upper 32 bits
                poly[z] = val;
            }
        }
        result.push(mat);
    }

    result
}

async fn handle_params(state: web::Data<Arc<ServerState>>) -> HttpResponse {
    let resp = serde_json::json!({
        "num_items": state.num_items,
        "tile_size": state.tile_size,
        "ypir_params": state.ypir_params_json,
        "setup_bytes": state.setup_bytes,
        "query_bytes": state.query_bytes,
        "response_bytes": state.response_bytes,
        "instances": state.instances,
        "rlwe_q_prime_1": state.rlwe_q_prime_1,
        "rlwe_q_prime_2": state.rlwe_q_prime_2,
    });
    HttpResponse::Ok().json(resp)
}

async fn handle_setup(
    state: web::Data<Arc<ServerState>>,
    body: web::Bytes,
) -> HttpResponse {
    let uuid = uuid::Uuid::new_v4().to_string();
    info!("Setup: {} ({} bytes)", uuid, body.len());

    let params = state.params;
    let condensed_row_1s = deserialize_pub_params(params, &body);

    if condensed_row_1s.len() != params.poly_len_log2 {
        return HttpResponse::BadRequest().body(format!(
            "Expected {} expansion params, got {}",
            params.poly_len_log2,
            condensed_row_1s.len()
        ));
    }

    let session = SessionData {
        pack_pub_params_row_1s: condensed_row_1s,
    };

    state.sessions.lock().unwrap().insert(uuid.clone(), session);
    HttpResponse::Ok().body(uuid)
}

async fn handle_query_batch(
    state: web::Data<Arc<ServerState>>,
    body: web::Bytes,
) -> HttpResponse {
    let params = state.params;
    let db_rows = 1usize << (params.db_dim_1 + params.poly_len_log2);
    let db_rows_padded = params.db_rows_padded();

    // Parse: [UUID:36][count:u32LE][q0_bytes]...[qN_bytes]
    if body.len() < 40 {
        return HttpResponse::BadRequest().body("Request too short");
    }

    let uuid = match std::str::from_utf8(&body[..36]) {
        Ok(s) => s.to_string(),
        Err(_) => return HttpResponse::BadRequest().body("Invalid UUID"),
    };

    let count = u32::from_le_bytes(body[36..40].try_into().unwrap()) as usize;
    let query_byte_size = state.query_bytes;

    let expected_len = 36 + 4 + count * query_byte_size;
    if body.len() < expected_len {
        return HttpResponse::BadRequest().body(format!(
            "Expected {} bytes for {} queries, got {}",
            expected_len, count, body.len()
        ));
    }

    info!("Query batch: UUID={}, count={}", uuid, count);

    // Clone session data out of the lock so we don't hold it during computation
    let pack_pub_params_row_1s = {
        let sessions = state.sessions.lock().unwrap();
        match sessions.get(&uuid) {
            Some(s) => s.pack_pub_params_row_1s.clone(),
            None => return HttpResponse::NotFound().body("Session not found"),
        }
    };

    let t_batch = Instant::now();

    // Process all queries in parallel using rayon
    let query_results: Vec<Vec<u8>> = (0..count)
        .into_par_iter()
        .map(|qi| {
            let q_start = 40 + qi * query_byte_size;
            let q_end = q_start + query_byte_size;
            let query_data = &body[q_start..q_end];

            // Deserialize CRT-packed query (u64 LE values) into 64-byte aligned memory
            // (required by AVX-512 _mm512_load_si512 in the YPIR kernel)
            let mut packed_query = AlignedMemory64::new(db_rows_padded);
            for i in 0..db_rows {
                let byte_start = i * 8;
                if byte_start + 8 <= query_data.len() {
                    packed_query.as_mut_slice()[i] = u64::from_le_bytes(
                        query_data[byte_start..byte_start + 8].try_into().unwrap()
                    );
                }
            }

            let t0 = Instant::now();
            let responses = state.y_server.perform_online_computation_simplepir(
                packed_query.as_slice(),
                &state.offline_vals,
                &[pack_pub_params_row_1s.as_slice()],
                None,
            );
            let elapsed = t0.elapsed();
            info!("Query {} computed in {:?}", qi, elapsed);

            // Serialize response: concatenated modulus-switched RLWE ciphertexts
            let mut out = Vec::new();
            for ct_bytes in &responses {
                out.extend_from_slice(ct_bytes);
            }
            out
        })
        .collect();

    info!("Batch of {} queries computed in {:?}", count, t_batch.elapsed());

    let mut all_responses = Vec::new();
    for r in query_results {
        all_responses.extend_from_slice(&r);
    }

    HttpResponse::Ok()
        .content_type("application/octet-stream")
        .body(all_responses)
}

fn build_ypir_params_json(params: &Params) -> String {
    // Serialize spiral-rs Params to JSON format that the WASM client can parse
    serde_json::json!({
        "n": params.n,
        "nu_1": params.db_dim_1,
        "nu_2": params.db_dim_2,
        "p": params.pt_modulus,
        "q2_bits": params.q2_bits,
        "t_gsw": params.t_gsw,
        "t_conv": params.t_conv,
        "t_exp_left": params.t_exp_left,
        "t_exp_right": params.t_exp_right,
        "instances": params.instances,
        "db_item_size": params.db_item_size,
        "moduli": params.moduli.iter().map(|m| m.to_string()).collect::<Vec<_>>(),
        "noise_width": params.noise_width,
        "poly_len": params.poly_len,
    }).to_string()
}

#[actix_web::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    let item_size_bits = cli.tile_size * 8;
    info!("Computing YPIR+SP params for {} items, {} bits each...", cli.num_tiles, item_size_bits);

    let params = params_for_scenario_simplepir(cli.num_tiles, item_size_bits);
    let params: &'static Params = Box::leak(Box::new(params));

    let db_rows = 1usize << (params.db_dim_1 + params.poly_len_log2);
    let db_cols = params.instances * params.poly_len;
    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();

    info!("Params: poly_len={}, nu_1={}, instances={}, pt_modulus={}",
        params.poly_len, params.db_dim_1, params.instances, params.pt_modulus);
    info!("DB dimensions: {} rows x {} cols", db_rows, db_cols);
    info!("q_prime_1={}, q_prime_2={}", rlwe_q_prime_1, rlwe_q_prime_2);

    // Compute response ciphertext size
    let q_1_bits = (rlwe_q_prime_2 as f64).log2().ceil() as usize;
    let q_2_bits = (rlwe_q_prime_1 as f64).log2().ceil() as usize;
    let ct_bits = (q_1_bits + q_2_bits) * params.poly_len;
    let ct_bytes = (ct_bits + 7) / 8;
    let response_bytes = params.instances * ct_bytes;
    let query_bytes = db_rows * 8; // CRT-packed u64s
    let setup_bytes = params.poly_len_log2 * params.t_exp_left * params.poly_len * 8;

    info!("Per-query: setup={} B, query={} B, response={} B", setup_bytes, query_bytes, response_bytes);

    // Load database
    info!("Loading database from {}...", cli.database);
    let db_data = load_db_as_u16_iter(&cli.database, params, cli.num_tiles, cli.tile_size);

    info!("Creating YPIR server...");
    let t0 = Instant::now();
    let y_server: &'static YServer<'static, u16> = Box::leak(Box::new(
        YServer::<u16>::new(params, db_data.into_iter(), true, false, true),
    ));
    info!("Server created in {:?}", t0.elapsed());

    // Offline precomputation
    info!("Running offline precomputation (this may take a while)...");
    let t0 = Instant::now();
    let offline_vals = y_server.perform_offline_precomputation_simplepir(None);
    info!("Offline precomputation done in {:?}", t0.elapsed());

    let ypir_params_json = build_ypir_params_json(params);

    let state = Arc::new(ServerState {
        params,
        y_server,
        offline_vals,
        sessions: Mutex::new(HashMap::new()),
        setup_bytes,
        query_bytes,
        response_bytes,
        num_items: cli.num_tiles,
        instances: params.instances,
        tile_size: cli.tile_size,
        ypir_params_json,
        rlwe_q_prime_1,
        rlwe_q_prime_2,
    });

    info!("Starting HTTP server on port {}...", cli.port);

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .app_data(web::PayloadConfig::new(200 * 1024 * 1024))
            .route("/api/params", web::get().to(handle_params))
            .route("/api/setup", web::post().to(handle_setup))
            .route("/api/query-batch", web::post().to(handle_query_batch))
    })
    .bind(("0.0.0.0", cli.port))
    .context("Failed to bind HTTP server")?
    .run()
    .await
    .context("HTTP server error")?;

    Ok(())
}
