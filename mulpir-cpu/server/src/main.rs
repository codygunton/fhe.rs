#![allow(missing_docs)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use actix_web::{middleware, web, App, HttpResponse, HttpServer};
use anyhow::Context;
use clap::Parser;
use fhe::bfv::{
    BfvParameters, BfvParametersBuilder, Ciphertext, Encoding, EvaluationKey,
    Plaintext, RelinearizationKey,
};
use fhe_traits::{DeserializeParametrized, FheEncoder, Serialize};
use fhe_util::transcode_from_bytes;
use memmap2::Mmap;
use tracing::{error, info};
use uuid::Uuid;

// ── Constants ─────────────────────────────────────────────────────────────
const DEGREE: usize = 8192;
const PLAINTEXT_MODULUS: u64 = (1 << 20) + (1 << 19) + (1 << 17) + (1 << 16) + (1 << 14) + 1;
const MODULI_SIZES: [usize; 3] = [50, 55, 55];
const BITS_PER_COEFF: usize = 20;

// ── CLI ───────────────────────────────────────────────────────────────────
#[derive(Parser)]
#[command(name = "mulpir-cpu-server")]
struct Cli {
    #[arg(long)]
    database: String,
    #[arg(long)]
    tile_mapping: String,
    #[arg(long)]
    num_tiles: usize,
    #[arg(long, default_value_t = 20480)]
    tile_size: usize,
    #[arg(long, default_value_t = 8081)]
    port: u16,
}

// ── Session ───────────────────────────────────────────────────────────────
struct SessionKeys {
    ek: EvaluationKey,
    rk: RelinearizationKey,
}

// ── Server state ──────────────────────────────────────────────────────────
struct ServerState {
    params: Arc<BfvParameters>,
    db: Vec<Plaintext>,
    dim1: usize,
    dim2: usize,
    num_tiles: usize,
    tile_size: usize,
    tile_mapping_json: String,
    sessions: RwLock<HashMap<String, SessionKeys>>,
}

// ── Helpers ───────────────────────────────────────────────────────────────
fn compute_dims(num_tiles: usize, tile_size: usize) -> (usize, usize, usize, usize) {
    let elements_per_pt =
        ((PLAINTEXT_MODULUS.ilog2() as usize * DEGREE) / (tile_size * 8)).max(1);
    let num_rows = num_tiles.div_ceil(elements_per_pt);
    let dim1 = (num_rows as f64).sqrt().ceil() as usize;
    let dim2 = num_rows.div_ceil(dim1);
    let expansion_level = (dim1 + dim2).next_power_of_two().ilog2() as usize;
    (dim1, dim2, expansion_level, elements_per_pt)
}

fn load_db(
    params: &Arc<BfvParameters>,
    raw: &[u8],
    num_tiles: usize,
    tile_size: usize,
) -> Vec<Plaintext> {
    let (dim1, dim2, _, elements_per_pt) = compute_dims(num_tiles, tile_size);
    let total_slots = dim1 * dim2;

    (0..total_slots)
        .map(|slot| {
            let mut slot_bytes = vec![0u8; elements_per_pt * tile_size];
            for elem in 0..elements_per_pt {
                let tile_idx = slot * elements_per_pt + elem;
                if tile_idx < num_tiles {
                    let src = &raw[tile_idx * tile_size..(tile_idx + 1) * tile_size];
                    let dst = &mut slot_bytes[elem * tile_size..(elem + 1) * tile_size];
                    dst.copy_from_slice(src);
                }
            }
            let coeff_values = transcode_from_bytes(&slot_bytes, BITS_PER_COEFF);
            Plaintext::try_encode(&coeff_values, Encoding::poly_at_level(1), params)
                .expect("encode plaintext")
        })
        .collect()
}

fn process_query(
    params: &Arc<BfvParameters>,
    ek: &EvaluationKey,
    rk: &RelinearizationKey,
    db: &[Plaintext],
    dim1: usize,
    dim2: usize,
    query_bytes: &[u8],
) -> anyhow::Result<Vec<u8>> {
    let query = Ciphertext::from_bytes(query_bytes, params)?;
    let expanded = ek.expands(&query, dim1 + dim2)?;

    let mut result = fhe::bfv::Ciphertext::zero(params);
    for j in 0..dim2 {
        let col_ct = fhe::bfv::dot_product_scalar(
            expanded[..dim1].iter(),
            db.iter().skip(j).step_by(dim2),
        )?;
        result += &(&col_ct * &expanded[dim1 + j]);
    }

    rk.relinearizes(&mut result)?;
    result.switch_to_level(result.max_switchable_level())?;
    Ok(result.to_bytes())
}

// ── Endpoints ─────────────────────────────────────────────────────────────

// POST /api/setup
// Body: [galois_key_len: u32 LE][galois_key_bytes][relin_key_bytes]
// Response: 36-char UUID text/plain
async fn setup(state: web::Data<Arc<ServerState>>, body: web::Bytes) -> HttpResponse {
    if body.len() < 4 {
        return HttpResponse::BadRequest().body("body too short");
    }
    let galois_len = u32::from_le_bytes(
        body[..4]
            .try_into()
            .expect("slice to array conversion"),
    ) as usize;
    if body.len() < 4 + galois_len {
        return HttpResponse::BadRequest().body("body too short for galois key");
    }
    let galois_bytes = body[4..4 + galois_len].to_vec();
    let relin_bytes = body[4 + galois_len..].to_vec();

    let params = state.params.clone();
    let result = web::block(move || {
        let ek = EvaluationKey::from_bytes(&galois_bytes, &params)?;
        let rk = RelinearizationKey::from_bytes(&relin_bytes, &params)?;
        Ok::<_, anyhow::Error>((ek, rk))
    })
    .await;

    match result {
        Ok(Ok((ek, rk))) => {
            let uuid = Uuid::new_v4().to_string();
            state
                .sessions
                .write()
                .expect("sessions RwLock poisoned")
                .insert(uuid.clone(), SessionKeys { ek, rk });
            info!(uuid = %uuid, "stored session keys");
            HttpResponse::Ok()
                .content_type("text/plain")
                .body(uuid)
        }
        Ok(Err(e)) => {
            error!("key deserialisation failed: {e}");
            HttpResponse::BadRequest().body(format!("key deserialisation failed: {e}"))
        }
        Err(e) => HttpResponse::InternalServerError().body(format!("{e}")),
    }
}

// POST /api/batch-query
// Body: [UUID: 36B][num_queries: u32 LE][q1_len: u32 LE][q1_bytes]...
// Response: [num_responses: u32 LE][r1_len: u32 LE][r1_bytes]...
async fn batch_query(state: web::Data<Arc<ServerState>>, body: web::Bytes) -> HttpResponse {
    const UUID_LEN: usize = 36;
    if body.len() < UUID_LEN + 4 {
        return HttpResponse::BadRequest().body("body too short");
    }
    let uuid = match std::str::from_utf8(&body[..UUID_LEN]) {
        Ok(s) => s.to_string(),
        Err(_) => return HttpResponse::BadRequest().body("invalid UUID"),
    };

    let num_queries =
        u32::from_le_bytes(body[UUID_LEN..UUID_LEN + 4].try_into().expect("4 bytes")) as usize;
    let mut queries: Vec<Vec<u8>> = Vec::with_capacity(num_queries);
    let mut off = UUID_LEN + 4;
    for _ in 0..num_queries {
        if off + 4 > body.len() {
            break;
        }
        let q_len = u32::from_le_bytes(body[off..off + 4].try_into().expect("4 bytes")) as usize;
        off += 4;
        if off + q_len > body.len() {
            break;
        }
        queries.push(body[off..off + q_len].to_vec());
        off += q_len;
    }

    // Get serialized keys from session
    let (ek_bytes, rk_bytes) = {
        let sessions = state.sessions.read().expect("sessions RwLock poisoned");
        match sessions.get(&uuid) {
            None => {
                return HttpResponse::NotFound().body(format!("unknown session: {uuid}"));
            }
            Some(s) => (s.ek.to_bytes(), s.rk.to_bytes()),
        }
    };

    let state_clone = state.clone();
    let result = web::block(move || {
        let params = &state_clone.params;
        let ek = EvaluationKey::from_bytes(&ek_bytes, params)?;
        let rk = RelinearizationKey::from_bytes(&rk_bytes, params)?;
        let responses: anyhow::Result<Vec<Vec<u8>>> = queries
            .iter()
            .map(|q| {
                process_query(
                    params,
                    &ek,
                    &rk,
                    &state_clone.db,
                    state_clone.dim1,
                    state_clone.dim2,
                    q,
                )
            })
            .collect();
        responses
    })
    .await;

    match result {
        Ok(Ok(responses)) => {
            let mut out = Vec::new();
            out.extend_from_slice(&(responses.len() as u32).to_le_bytes());
            for r in &responses {
                out.extend_from_slice(&(r.len() as u32).to_le_bytes());
                out.extend_from_slice(r);
            }
            HttpResponse::Ok()
                .content_type("application/octet-stream")
                .body(out)
        }
        Ok(Err(e)) => {
            error!("query processing failed: {e}");
            HttpResponse::InternalServerError().body(format!("{e}"))
        }
        Err(e) => HttpResponse::InternalServerError().body(format!("{e}")),
    }
}

// GET /api/params → JSON
async fn api_params(state: web::Data<Arc<ServerState>>) -> HttpResponse {
    let (_, _, expansion_level, elements_per_pt) =
        compute_dims(state.num_tiles, state.tile_size);
    HttpResponse::Ok().json(serde_json::json!({
        "num_tiles": state.num_tiles,
        "tile_size": state.tile_size,
        "dim1": state.dim1,
        "dim2": state.dim2,
        "expansion_level": expansion_level,
        "elements_per_plaintext": elements_per_pt,
    }))
}

// GET /api/tile-mapping → JSON
async fn tile_mapping(state: web::Data<Arc<ServerState>>) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/json")
        .body(state.tile_mapping_json.clone())
}

// ── main ──────────────────────────────────────────────────────────────────
#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let params = BfvParametersBuilder::new()
        .set_degree(DEGREE)
        .set_plaintext_modulus(PLAINTEXT_MODULUS)
        .set_moduli_sizes(&MODULI_SIZES)
        .build_arc()
        .context("failed to build BFV parameters")?;

    let (dim1, dim2, expansion_level, elements_per_pt) =
        compute_dims(cli.num_tiles, cli.tile_size);

    info!(
        num_tiles = cli.num_tiles,
        tile_size = cli.tile_size,
        dim1,
        dim2,
        expansion_level,
        elements_per_pt,
        "computed PIR dimensions"
    );

    info!(database = %cli.database, "loading database");

    let file = std::fs::File::open(&cli.database)
        .with_context(|| format!("failed to open database: {}", cli.database))?;
    // SAFETY: file must not be modified while mmap is live
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap database: {}", cli.database))?;

    let db = load_db(&params, mmap.as_ref(), cli.num_tiles, cli.tile_size);
    info!(plaintexts = db.len(), "database loaded");

    let tile_mapping_json = std::fs::read_to_string(&cli.tile_mapping)
        .with_context(|| format!("failed to read tile mapping: {}", cli.tile_mapping))?;

    let state = Arc::new(ServerState {
        params,
        db,
        dim1,
        dim2,
        num_tiles: cli.num_tiles,
        tile_size: cli.tile_size,
        tile_mapping_json,
        sessions: RwLock::new(HashMap::new()),
    });

    info!(port = cli.port, "starting MulPIR-CPU server");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .app_data(web::JsonConfig::default().limit(200 * 1024 * 1024))
            .app_data(web::PayloadConfig::new(200 * 1024 * 1024))
            .wrap(middleware::Logger::default())
            .route("/api/setup", web::post().to(setup))
            .route("/api/batch-query", web::post().to(batch_query))
            .route("/api/params", web::get().to(api_params))
            .route("/api/tile-mapping", web::get().to(tile_mapping))
    })
    .bind(("0.0.0.0", cli.port))
    .with_context(|| format!("failed to bind port {}", cli.port))?
    .run()
    .await
    .context("server exited with error")
}
