use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, RwLock};

use actix_web::{middleware, web, App, HttpResponse, HttpServer};
use anyhow::Context;
use clap::Parser;
use memmap2::Mmap;
use spiral_rs::client::{PublicParameters, Query};
use spiral_rs::params::Params;
use spiral_rs::server::{load_db_from_seek, process_query};
use spiral_rs::util::params_from_json;
use tracing::{error, info};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "spiral-cpu-server")]
struct Cli {
    /// Path to the flat tiles binary database file.
    #[arg(long)]
    database: String,

    /// Path to the tile mapping JSON file.
    #[arg(long)]
    tile_mapping: String,

    /// Number of tiles in the database.
    #[arg(long)]
    num_tiles: usize,

    /// Size of each tile in bytes.
    #[arg(long, default_value_t = 20480)]
    tile_size: usize,

    /// Port to listen on.
    #[arg(long, default_value_t = 8081)]
    port: u16,
}

/// Shared server state accessible from all request handlers.
struct ServerState {
    /// Spiral-rs parameters (leaked to `'static` at startup).
    params: &'static Params,
    /// The preprocessed database as a flat `Vec<u64>` (Send + Sync).
    db: Vec<u64>,
    /// Map from UUID string to serialized `PublicParameters` bytes.
    pub_params: RwLock<HashMap<String, Vec<u8>>>,
    /// Number of tiles in the database.
    num_tiles: usize,
    /// Size of each tile in bytes.
    tile_size: usize,
    /// Contents of the tile mapping JSON file.
    tile_mapping_json: String,
    /// JSON string used to construct the Spiral params.
    params_json: String,
}

/// Choose Spiral params JSON for a given number of tiles and tile size.
///
/// Parameters follow Blyss's production v1 scheme:
///   - version=1, p=256, t_exp_left=t_exp_right=5 — triggers the right-expansion
///     skip in spiral-rs (expansion_right_sz = 0), cutting setup size ~7x.
///   - instances is the minimum needed to fit `tile_size` bytes per item:
///     each chunk holds `poly_len * log2(p) / 8 = 2048` bytes (with p=256),
///     and there are `instances * n * n` chunks per item, so
///     `instances = ceil(tile_size / (n * n * 2048))`.
///   - nu_1=9 (large left dimension) matches the Blyss v1 production shape;
///     nu_2 is chosen to cover at least `num_tiles` total items.
fn select_params_json(num_tiles: usize, tile_size: usize) -> String {
    // With p=256 (8 bits/coeff) and poly_len=2048: 2048 bytes per chunk.
    // chunks = instances * n^2 = instances * 4
    let bytes_per_chunk = 2048usize; // poly_len * log2(p) / 8
    let chunks_needed = tile_size.div_ceil(bytes_per_chunk);
    let instances = chunks_needed.div_ceil(4); // n^2 = 4

    // nu_1=9 (left dimension = 512 rows) matches Blyss v1 production shape.
    // nu_2 is chosen so total capacity 2^(9+nu_2) >= num_tiles.
    let nu_1 = 9usize;
    let nu_2 = if num_tiles <= (1 << (nu_1 + 2)) {
        2 // 2^11 = 2048 items
    } else if num_tiles <= (1 << (nu_1 + 4)) {
        4 // 2^13 = 8192 items
    } else {
        6 // 2^15 = 32768 items
    };

    serde_json::json!({
        "n": 2,
        "nu_1": nu_1,
        "nu_2": nu_2,
        "p": 256,
        "q2_bits": 22,
        "t_gsw": 7,
        "t_conv": 3,
        "t_exp_left": 5,
        "t_exp_right": 5,
        "instances": instances,
        "db_item_size": tile_size,
        "version": 1
    })
    .to_string()
}

/// `POST /api/setup`
///
/// Body: raw bytes of a serialized `PublicParameters` (from the spiral-wasm
/// client's `generate_keys()`).
///
/// Response: UUID string (`text/plain`) that identifies this session's keys.
async fn setup(state: web::Data<Arc<ServerState>>, body: web::Bytes) -> HttpResponse {
    let params = state.params;
    // Deserialize to validate the bytes, then re-serialize for storage so that
    // we avoid holding a `PublicParameters<'a>` with a lifetime tied to `params`.
    let pp = PublicParameters::deserialize(params, &body);
    let serialized = pp.serialize();
    let uuid = Uuid::new_v4().to_string();
    {
        let mut map = state
            .pub_params
            .write()
            .expect("pub_params RwLock poisoned");
        map.insert(uuid.clone(), serialized);
    }
    info!(uuid = %uuid, "stored public params for session");
    HttpResponse::Ok()
        .content_type("text/plain")
        .body(uuid)
}

/// `POST /api/private-read`
///
/// Body: `[UUID: 36 ASCII bytes][query bytes]`
///
/// Response: raw response bytes from `process_query`.
async fn private_read(
    state: web::Data<Arc<ServerState>>,
    body: web::Bytes,
) -> HttpResponse {
    const UUID_LEN: usize = 36;
    if body.len() < UUID_LEN {
        return HttpResponse::BadRequest()
            .body("body too short: need a 36-byte UUID prefix");
    }

    let uuid = match std::str::from_utf8(&body[..UUID_LEN]) {
        Ok(s) => s.to_string(),
        Err(_) => return HttpResponse::BadRequest().body("UUID prefix is not valid UTF-8"),
    };
    let query_bytes = body[UUID_LEN..].to_vec();

    let params = state.params;

    let pp_bytes = {
        let map = state.pub_params.read().expect("pub_params RwLock poisoned");
        match map.get(&uuid) {
            Some(b) => b.clone(),
            None => {
                return HttpResponse::NotFound()
                    .body(format!("unknown session UUID: {uuid}"));
            }
        }
    };

    let state_clone = state.clone();
    let result = web::block(move || {
        let pp = PublicParameters::deserialize(params, &pp_bytes);
        let query = Query::deserialize(params, &query_bytes);
        process_query(params, &pp, &query, &state_clone.db)
    })
    .await;

    match result {
        Ok(resp) => HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body(resp),
        Err(err) => {
            error!(error = %err, "process_query failed");
            HttpResponse::InternalServerError().body("query processing failed")
        }
    }
}

/// `GET /api/params`
///
/// Returns a JSON object describing the server's Spiral parameters and
/// database dimensions.
async fn api_params(state: web::Data<Arc<ServerState>>) -> HttpResponse {
    let params = state.params;
    let body = serde_json::json!({
        "num_tiles": state.num_tiles,
        "tile_size": state.tile_size,
        "spiral_params": state.params_json,
        "setup_bytes": params.setup_bytes(),
        "query_bytes": params.query_bytes(),
        "num_items": params.num_items(),
    });
    HttpResponse::Ok().json(body)
}

/// `GET /api/tile-mapping`
///
/// Returns the raw tile-mapping JSON loaded at startup.
async fn tile_mapping(state: web::Data<Arc<ServerState>>) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/json")
        .body(state.tile_mapping_json.clone())
}

#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let params_json = select_params_json(cli.num_tiles, cli.tile_size);
    info!(params = %params_json, "selected Spiral params");

    let params = Box::new(params_from_json(&params_json));
    // SAFETY: `params` is never freed while the server is running.
    let params: &'static Params = Box::leak(params);

    info!(
        database = %cli.database,
        num_tiles = cli.num_tiles,
        tile_size = cli.tile_size,
        "loading database"
    );

    let file =
        std::fs::File::open(&cli.database).with_context(|| {
            format!("failed to open database file: {}", cli.database)
        })?;
    // SAFETY: the file must not be modified while the mmap is live.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap database file: {}", cli.database))?;
    let mut cursor = Cursor::new(mmap.as_ref());
    let db_aligned = load_db_from_seek(params, &mut cursor);
    let db: Vec<u64> = db_aligned.as_slice().to_vec();

    info!(words = db.len(), "database loaded");

    let tile_mapping_json = std::fs::read_to_string(&cli.tile_mapping)
        .with_context(|| format!("failed to read tile mapping file: {}", cli.tile_mapping))?;

    let state = Arc::new(ServerState {
        params,
        db,
        pub_params: RwLock::new(HashMap::new()),
        num_tiles: cli.num_tiles,
        tile_size: cli.tile_size,
        tile_mapping_json,
        params_json: params_json.clone(),
    });

    info!(
        port = cli.port,
        num_items = params.num_items(),
        setup_bytes = params.setup_bytes(),
        query_bytes = params.query_bytes(),
        "starting Spiral-CPU server"
    );

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .app_data(web::JsonConfig::default().limit(200 * 1024 * 1024))
            .app_data(web::PayloadConfig::new(200 * 1024 * 1024))
            .wrap(middleware::Logger::default())
            .route("/api/setup", web::post().to(setup))
            .route("/api/private-read", web::post().to(private_read))
            .route("/api/params", web::get().to(api_params))
            .route("/api/tile-mapping", web::get().to(tile_mapping))
    })
    .bind(("0.0.0.0", cli.port))
    .with_context(|| format!("failed to bind to port {}", cli.port))?
    .run()
    .await
    .context("server exited with error")
}
