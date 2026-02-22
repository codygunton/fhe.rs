# Spiral-CPU Demo Implementation Plan

## Executive Summary

We are adding a second PIR backend to the map demo — **Spiral-CPU** — alongside the existing
**MulPIR-GPU** implementation, so we can compare the two schemes on identical data with an
identical UX. The goal is an apples-to-apples performance comparison: same MapLibre frontend,
same tile database, same batching and caching optimizations, but swapped-out PIR engine.

The key architectural decision is **code sharing**. The PIR-independent map layer — tile cache,
viewport coalescing, speculative prefetch, gzip decoding, MapLibre setup — must be shared so
both demos reflect the same optimizations and a change to one propagates to the other. Only the
PIR-specific wiring (key setup, query generation, response decryption, backend transport)
differs between the two.

### Data Flow

```
Browser
  ├── [Shared] MapLibre → viewport coalescing → LRU cache check
  │                    └── TileBatchDispatcher (shared, injectable PIR backend)
  │                          ├── [MulPIR] BFV WASM (fhe-wasm) → binary over TCP → Flask → GPU server
  │                          └── [Spiral] Spiral WASM (spiral-wasm) → HTTP JSON → Flask → CPU server
  └── [Shared] Tile decoder ([u32 len][gzip][padding] → PBF)
```

### Directory Layout After This Plan

```
fhe.rs/
├── pir-map-shared/           NEW — PIR-agnostic JS modules + Python proxy base
│   ├── frontend/
│   │   ├── tile-cache.js     LRUTileCache (extracted from mulpir-gpu)
│   │   ├── tile-batch.js     TileBatchDispatcher with injectable PIR backend
│   │   ├── tile-decoder.js   [u32 len][gzip][padding] → Uint8Array
│   │   └── map-setup.js      MapLibre init, pir:// protocol handler
│   └── proxy/
│       └── common.py         Shared Flask helpers (port kill, tile-mapping, static serve)
│
├── mulpir-gpu/               MODIFIED — app.js refactored to use shared modules
│   └── demo/frontend/app.js  Now imports from ../../pir-map-shared/frontend/
│
├── spiral-cpu/               NEW
│   ├── server/               Rust HTTP server (spiral-rs + actix-web)
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   ├── demo/
│   │   ├── frontend/
│   │   │   ├── index.html    (copy of mulpir-gpu version, adjusted title/labels)
│   │   │   ├── style.css     (shared copy)
│   │   │   ├── app.js        Spiral-specific init + shared modules
│   │   │   └── pkg/          symlink → crates/spiral-wasm/pkg
│   │   ├── proxy/
│   │   │   └── server.py     Flask proxy (imports common.py, HTTP forward)
│   │   ├── tiles/            symlink → ../../mulpir-gpu/demo/tiles/
│   │   └── scripts/
│   │       └── run_demo.sh
│   └── run_demo.sh
│
└── crates/
    └── spiral-wasm/          NEW — WASM wrapper for spiral-rs client
        ├── Cargo.toml
        └── src/lib.rs
```

### Expected Outcomes

- Both demos run independently: `./mulpir-gpu/run_demo.sh` and `./spiral-cpu/run_demo.sh`
- Both open at `http://localhost:800X` showing the same map, same tiles
- Latency and bandwidth differences are directly attributable to the PIR scheme
- Changes to tile batching, caching, or prefetch logic apply equally to both without duplication

---

## Goals & Objectives

### Primary Goals

- Running Spiral-CPU demo on the same tile data as MulPIR-GPU
- Shared PIR-agnostic code between both demos (zero duplication of map/cache/batch logic)
- Comparable baseline: identical batching window (50ms), identical LRU cache, identical prefetch

### Secondary Objectives

- Keep spiral-cpu as a self-contained directory (buildable independently)
- Avoid modifying `sdk/` (treat it as a read-only vendored dependency)
- `spiral-wasm` is a proper Cargo workspace member, built with `wasm-pack` like `fhe-wasm`

---

## Solution Overview

### Approach

**Shared JS layer** (`pir-map-shared/frontend/`): Extract the four PIR-agnostic modules from
`mulpir-gpu/demo/frontend/app.js`. The key abstraction is `TileBatchDispatcher`, which takes
a `pirBackend` object as a constructor argument. Each demo provides its own backend object
implementing `{ setup(), createQuery(slot), decryptResponse(bytes, slot) }`. The dispatcher
and everything above it is shared.

**Spiral WASM** (`crates/spiral-wasm/`): Thin `wasm-bindgen` wrapper around `spiral-rs`'s
`Client` struct. Exposes: `SpiralClient.new(paramsJson)`, `.generateKeys()`,
`.generateQuery(idx)`, `.decodeResponse(bytes)`. Built with `wasm-pack --target web`.
Parallel to `fhe-wasm`.

**Spiral HTTP Server** (`spiral-cpu/server/`): Rust binary using `actix-web` and `spiral-rs`
(with `server` feature). At startup, memory-maps `tiles.bin` and initializes the spiral-rs
database directly (no loader script). Exposes HTTP endpoints that the proxy forwards to.
Tile slot `i` maps directly to spiral-rs row `i` (index-based, not KV — no hash collisions,
decrypted response is the raw 20480-byte slot, same format as MulPIR).

**Spiral proxy** (`spiral-cpu/demo/proxy/server.py`): Same Flask structure as mulpir-gpu
proxy, but backend communication is HTTP (no binary TCP). Imports shared `common.py` for
the parts that are identical.

### Key Components

1. **pir-map-shared**: Canonical home for all map/tile code that doesn't know about BFV or Spiral
2. **crates/spiral-wasm**: WASM client, mirrors fhe-wasm structure exactly
3. **spiral-cpu/server**: Self-contained Rust HTTP server; reads tiles.bin, runs spiral-rs
4. **spiral-cpu/demo/frontend/app.js**: Spiral-specific glue (setup, key gen, query, decrypt)
5. **spiral-cpu/demo/proxy**: Thin HTTP-forward proxy using shared common.py

### Architecture Diagram

```
mulpir-gpu/demo/frontend/app.js        spiral-cpu/demo/frontend/app.js
  BFV backend object                     Spiral backend object
       |                                      |
       └─────────────┬────────────────────────┘
                     ▼
          pir-map-shared/frontend/
            tile-cache.js (LRUTileCache)
            tile-batch.js (TileBatchDispatcher)
            tile-decoder.js (slot → gzip PBF)
            map-setup.js (MapLibre + pir:// protocol)
```

### Wire Formats

**MulPIR-GPU** (unchanged):
```
Setup:  POST /api/setup-galois-key + POST /api/setup-relin-key  (raw bytes → TCP)
Query:  POST /api/batch-query  [u32 num][u32 size][query bytes]... → binary response
```

**Spiral-CPU** (new):
```
Setup:  POST /api/setup  raw bytes(PublicParameters) → text/plain UUID
Query:  POST /api/private-read  raw bytes: [UUID(36)][query bytes] → raw bytes response
Params: GET /api/params  → JSON (num_tiles, tile_size, item_size, spiral_params_json)
```

Both decoded responses have the same 20480-byte slot format: `[u32 data_len LE][gzip PBF][padding]`.

---

## Implementation Tasks

### CRITICAL IMPLEMENTATION RULES

1. **NO PLACEHOLDER CODE**: Every implementation must be production-ready.
2. **No KV layer**: Use index-based spiral-rs access (`generate_query(slot_idx)`), not the
   Blyss KV abstraction (`privateRead("z/x/y")`). Avoids hash collisions and keeps the
   decrypted format identical to MulPIR.
3. **Shared tiles**: `spiral-cpu/demo/tiles` is a symlink, never a copy.
4. **Workspace membership**: `spiral-cpu/server` and `crates/spiral-wasm` must be in `Cargo.toml`.
5. **Param validation**: Spiral params for `db_item_size=20480` must be validated to work
   with `num_items` covering our largest dataset (4213 tiles → use `db_dim_1=6, db_dim_2=7`
   for 8192-item capacity).

### Visual Dependency Tree

```
pir-map-shared/                       (Task A1)
  frontend/tile-cache.js
  frontend/tile-batch.js
  frontend/tile-decoder.js
  frontend/map-setup.js
  proxy/common.py

crates/spiral-wasm/                   (Task A2)
  Cargo.toml
  src/lib.rs

spiral-cpu/server/                    (Task A3)
  Cargo.toml
  src/main.rs

spiral-cpu/demo/
  proxy/server.py                     (Task A4 — depends on pir-map-shared/proxy/common.py)
  tiles/                              (Task A5 — symlink only)
  frontend/                           (Task B1 — depends on A1, A2)
    index.html
    style.css
    app.js
    pkg/  → symlink to crates/spiral-wasm/pkg
  scripts/run_demo.sh                 (Task C1 — depends on A3, A4, B1)

spiral-cpu/run_demo.sh                (Task C1)

mulpir-gpu/demo/frontend/app.js       (Task A1 — refactored to import from shared)
mulpir-gpu/demo/proxy/server.py       (Task A1 — refactored to import from shared)
Cargo.toml (workspace)                (Task A2, A3 — add new members)
```

### Execution Plan

---

#### Group A: Foundation (all tasks execute in parallel)

- [ ] **Task A1**: Extract shared frontend/proxy code and refactor mulpir-gpu
  - **New files**:
    - `pir-map-shared/frontend/tile-cache.js` — Extract `LRUTileCache` class verbatim from
      `mulpir-gpu/demo/frontend/app.js` lines ~4-38. No changes to logic.
    - `pir-map-shared/frontend/tile-decoder.js` — Extract tile slot decoder: reads `[u32 data_len LE]`
      from first 4 bytes, returns `Uint8Array(data_len)` containing raw gzip bytes.
      Signature: `export function decodeSlot(slotBytes: Uint8Array): Uint8Array`
    - `pir-map-shared/frontend/tile-batch.js` — Refactor `TileBatchDispatcher` to accept a
      `pirBackend` object at construction time. The backend interface:
      ```javascript
      pirBackend = {
        isReady(): bool,
        createQuery(slotIndex): Uint8Array,        // encrypted query bytes
        decryptResponse(responseBytes, slotIndex): Uint8Array  // decrypted slot (20480B)
      }
      ```
      Dispatcher coalescing window, deduplication, and abort handling remain identical.
      The `executeBatch(queries)` method signature changes to call `pirBackend.createQuery(slot)`
      and submit to `pirBackend.submitBatch(queryList)` → response array.
      `pirBackend.submitBatch` handles the actual HTTP POST (scheme-specific).
    - `pir-map-shared/frontend/map-setup.js` — Extract MapLibre init: `initMap(center, minZoom,
      maxZoom, fetchTileFn)`. Sets up the `pir://` protocol handler and `addSource/addLayer` calls.
    - `pir-map-shared/proxy/common.py` — Extract shared Flask helpers from
      `mulpir-gpu/demo/proxy/server.py`:
      - `kill_port(port)` — kill processes on a port
      - `load_tile_mapping(tiles_dir)` — read tile_mapping.json
      - `compute_pir_params(num_tiles, tile_size)` — dimension calculation
      - `serve_frontend(app, frontend_dir)` — Flask static file routes
  - **Modified files**:
    - `mulpir-gpu/demo/frontend/app.js` — Import from `../../pir-map-shared/frontend/`.
      Replace inline `LRUTileCache`, `TileBatchDispatcher`, tile decoder, and MapLibre init
      with imports. BFV-specific code (key gen, galois/relin setup, query creation, decrypt)
      remains in this file as a `bfvBackend` object passed to `TileBatchDispatcher`.
    - `mulpir-gpu/demo/frontend/index.html` — Add `<script>` tags for shared modules
      (loaded before app.js): `tile-cache.js`, `tile-batch.js`, `tile-decoder.js`, `map-setup.js`
      from `../../pir-map-shared/frontend/` (relative URL path via symlinks or proxy route).
    - `mulpir-gpu/demo/proxy/server.py` — Import helpers from `common.py`. Remove duplicated
      logic. Keep MulPIR-specific: binary TCP client, GPU metrics endpoint.
  - **Note on static file serving for shared modules**: The Flask proxy must serve `pir-map-shared/frontend/`
    at URL path `/shared/` so both frontends can `import` or `<script src="/shared/...">` them.
    Add route to both proxies.

---

- [ ] **Task A2**: Create `crates/spiral-wasm/` WASM client
  - **File**: `crates/spiral-wasm/Cargo.toml`
    ```toml
    [package]
    name = "spiral-wasm"
    version = "0.1.0"
    edition.workspace = true

    [lib]
    crate-type = ["cdylib", "rlib"]

    [dependencies]
    spiral-rs = { path = "../../sdk/lib/spiral-rs" }
    wasm-bindgen = "0.2"
    getrandom = { version = "0.2", features = ["js"] }
    serde_json = "1.0"
    console_error_panic_hook = "0.1"
    js-sys = "0.3"
    ```
  - **File**: `crates/spiral-wasm/src/lib.rs`
    - `#[wasm_bindgen(start)]` init function (console_error_panic_hook)
    - `#[wasm_bindgen] pub struct SpiralClient { inner: spiral_rs::client::Client<'static>, params: &'static spiral_rs::params::Params }`
    - Methods exposed via `#[wasm_bindgen]`:
      - `SpiralClient::new(params_json: &str) -> SpiralClient` — deserialize params from JSON string
      - `fn generate_keys(&mut self) -> Vec<u8>` — call `client.generate_keys()`, serialize PublicParameters
      - `fn generate_query(&mut self, idx: usize) -> Vec<u8>` — call `client.generate_query(idx)`, serialize Query
      - `fn decode_response(&mut self, data: &[u8]) -> Vec<u8>` — call `client.decode_response(data)`
      - `fn query_bytes(&self) -> usize` — return `params.query_bytes()`
      - `fn setup_bytes(&self) -> usize` — return `params.setup_bytes()`
      - `fn num_items(&self) -> usize` — return `params.num_items()`
    - The `Params` lifetime issue: use `Box::leak` to create `'static` ref from runtime-loaded params
  - **Workspace update**: Add `"crates/spiral-wasm"` to `Cargo.toml` workspace members
  - **Build**: `wasm-pack build --target web --release` in `crates/spiral-wasm/`
  - **Output**: `crates/spiral-wasm/pkg/` (spiral_wasm.js + spiral_wasm_bg.wasm)

---

- [ ] **Task A3**: Create `spiral-cpu/server/` Rust HTTP server
  - **File**: `spiral-cpu/server/Cargo.toml`
    ```toml
    [package]
    name = "spiral-cpu-server"
    version = "0.1.0"
    edition = "2021"

    [[bin]]
    name = "spiral_server"
    path = "src/main.rs"

    [dependencies]
    spiral-rs = { path = "../../sdk/lib/spiral-rs", features = ["server"] }
    actix-web = { version = "4", features = ["macros"] }
    tokio = { version = "1", features = ["full"] }
    clap = { version = "4", features = ["derive"] }
    serde = { version = "1", features = ["derive"] }
    serde_json = "1"
    uuid = { version = "1", features = ["v4"] }
    rayon = "1"
    memmap2 = "0.9"
    log = "0.4"
    env_logger = "0.11"

    [profile.release]
    opt-level = 3
    ```
  - **File**: `spiral-cpu/server/src/main.rs`
    - **CLI args** (clap):
      ```
      --database <path>       Path to tiles.bin
      --tile-mapping <path>   Path to tile_mapping.json
      --num-tiles <N>         Number of PIR slots
      --tile-size <N>         Bytes per slot (default 20480)
      --port <N>              HTTP port (default 8081)
      ```
    - **Startup sequence**:
      1. Parse args
      2. Select Spiral params: `db_item_size = tile_size`, `db_dim_1 = 6, db_dim_2 = 7`
         (8192 capacity). For small datasets (≤128 tiles), use `db_dim_1=3, db_dim_2=4`.
         Expose a `select_params(num_tiles, item_size) -> &'static Params` helper.
      3. Memory-map tiles.bin
      4. Initialize spiral-rs database: call `spiral_rs::server::load_db_from_seek()` or
         directly construct the database array from mmap bytes
         (each 20480-byte slot → one spiral DB item)
      5. Start actix-web server
    - **State struct** (wrapped in `web::Data<Arc<...>>`):
      ```rust
      struct ServerState {
          params: &'static Params,
          db: Vec<AlignedMemory64>,        // spiral-rs database, read-only after init
          pub_params: RwLock<HashMap<String, PublicParameters<'static>>>,
          num_tiles: usize,
          tile_size: usize,
          tile_mapping_json: String,       // raw JSON string from tile_mapping.json
          params_json: String,             // spiral params as JSON for client
      }
      ```
    - **Endpoints**:
      - `POST /api/setup` — Body: raw bytes (PublicParameters serialized by spiral-wasm client).
        Deserialize via `PublicParameters::deserialize(params, &body)`.
        Store in `pub_params` map under new UUID v4. Return UUID as plain text.
        Max body: 10 MB.
      - `POST /api/private-read` — Body: raw bytes `[UUID(36 ASCII bytes)][query bytes]`.
        Extract UUID, look up pub_params. Deserialize query via `Query::deserialize(params, &query_bytes)`.
        Call `spiral_rs::server::process_query(params, &pub_params, &query, &db)` → `Vec<u8>`.
        Return raw bytes. Max body: 2 MB.
      - `GET /api/params` — Return JSON:
        ```json
        {
          "num_tiles": N,
          "tile_size": 20480,
          "spiral_params": "<params JSON string>",
          "setup_bytes": N,
          "query_bytes": N
        }
        ```
      - `GET /api/tile-mapping` — Return `tile_mapping_json` as `application/json`
    - **Rayon**: process_query is CPU-bound; actix runs it in `web::block(|| ...)` to avoid
      blocking the async runtime
  - **Workspace update**: Add `"spiral-cpu/server"` to `Cargo.toml` workspace members

---

- [ ] **Task A4**: Create `spiral-cpu/demo/proxy/server.py`
  - **File**: `spiral-cpu/demo/proxy/server.py`
    - Imports `common.py` from `pir-map-shared/proxy/` (via sys.path or relative import)
    - CLI args: `--spiral-port` (default 8081), `--port` (default 8002), `--tiles-dir`
    - Flask app with `MAX_CONTENT_LENGTH = 10 * 1024 * 1024`
    - Routes:
      - `GET /` → serve `frontend/index.html`
      - `GET /shared/<path>` → serve from `pir-map-shared/frontend/` (for shared JS modules)
      - `GET /<path>` → serve from `frontend/` static files
      - `POST /api/setup` → forward raw body to `http://localhost:{spiral_port}/api/setup`,
        return UUID text
      - `POST /api/private-read` → forward raw body, return raw response bytes
      - `GET /api/params` → forward to spiral server, return JSON
      - `GET /api/tile-mapping` → use `common.load_tile_mapping(tiles_dir)`, return JSON
      - `GET /api/metrics` → return CPU/memory stats (psutil): `{"cpu_percent": X, "memory_mb": Y}`
    - No TCP binary protocol (spiral server speaks HTTP directly)
  - **File**: `spiral-cpu/demo/proxy/requirements.txt`
    ```
    flask>=3.0
    requests>=2.31
    psutil>=5.9
    ```

---

- [ ] **Task A5**: Create directory structure and symlinks
  - `mkdir -p spiral-cpu/demo/{frontend,proxy,tiles,scripts}`
  - `ln -sf ../../mulpir-gpu/demo/tiles spiral-cpu/demo/tiles` (shared tile database)
  - `mkdir -p pir-map-shared/{frontend,proxy}`
  - Copy `mulpir-gpu/demo/frontend/style.css` → `spiral-cpu/demo/frontend/style.css`
  - Copy `mulpir-gpu/demo/frontend/index.html` → `spiral-cpu/demo/frontend/index.html`
    (adjust title to "Spiral-CPU PIR Map Demo", adjust metric labels from GPU to CPU)

---

#### Group B: Spiral Frontend (requires A1 and A2 to complete)

- [ ] **Task B1**: Create `spiral-cpu/demo/frontend/app.js`
  - Imports (via `<script>` in index.html, loaded before app.js):
    - `/shared/tile-cache.js` → `LRUTileCache`
    - `/shared/tile-batch.js` → `TileBatchDispatcher`
    - `/shared/tile-decoder.js` → `decodeSlot`
    - `/shared/map-setup.js` → `initMap`
    - `./pkg/spiral_wasm.js` → `init, SpiralClient`
  - **Initialization sequence** (replaces BFV key setup in mulpir-gpu app.js):
    ```javascript
    // 1. Load WASM
    await init();
    // 2. Fetch Spiral params from server
    const params = await fetch('/api/params').then(r => r.json());
    // 3. Create SpiralClient from params JSON
    const spiralClient = SpiralClient.new(params.spiral_params);
    // 4. Generate public parameters (slow, ~500ms-2s)
    showStatus('Generating keys...');
    const pubParamsBytes = spiralClient.generate_keys();
    // 5. POST to /api/setup → UUID
    const uuid = await fetch('/api/setup', {
      method: 'POST', body: pubParamsBytes,
      headers: {'Content-Type': 'application/octet-stream'}
    }).then(r => r.text());
    // 6. Build spiralBackend object for TileBatchDispatcher
    const spiralBackend = {
      isReady: () => true,
      createQuery: (slotIdx) => {
        const q = spiralClient.generate_query(slotIdx);
        // Prepend UUID (36 ASCII bytes) to query
        const full = new Uint8Array(36 + q.length);
        full.set(new TextEncoder().encode(uuid), 0);
        full.set(q, 36);
        return full;
      },
      decryptResponse: (responseBytes, slotIdx) => {
        return spiralClient.decode_response(responseBytes);
      },
      submitBatch: async (queryList, abortSignal) => {
        // Spiral does not multiplex queries — send one POST per query (or extend server later)
        // For now: send queries sequentially (batching window still applies for dedup/coalesce)
        return Promise.all(queryList.map(q =>
          fetch('/api/private-read', {
            method: 'POST', body: q,
            headers: {'Content-Type': 'application/octet-stream'},
            signal: abortSignal
          }).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))
        ));
      }
    };
    // 7. Fetch tile mapping
    const tileMapping = await fetch('/api/tile-mapping').then(r => r.json());
    // 8. Init map via shared module
    const tileCache = new LRUTileCache(500 * 1024 * 1024);
    const dispatcher = new TileBatchDispatcher(spiralBackend, { windowMs: 50 });
    initMap(tileMapping.center, tileMapping.min_zoom, tileMapping.max_zoom,
            (z, x, y) => fetchTile(z, x, y, tileMapping, tileCache, dispatcher));
    ```
  - **Tile fetch function**: same logic as mulpir-gpu (lookup slot from tileMapping, call
    `dispatcher.enqueue(slot)`, call `decodeSlot(decryptedBytes)` → gzip, decompress → PBF)
  - **Metrics panel**: Polls `/api/metrics` every 2s for CPU% and memory MB (no GPU stats)
  - **Symlink**: `spiral-cpu/demo/frontend/pkg` → `../../../crates/spiral-wasm/pkg`

---

#### Group C: Run Scripts (requires A3, A4, B1 to complete)

- [ ] **Task C1**: Create run scripts
  - **File**: `spiral-cpu/run_demo.sh`
    - Same structure as `mulpir-gpu/run_demo.sh`
    - ROOT = directory of this script (`spiral-cpu/`)
    - Builds spiral-wasm WASM if `ROOT/../crates/spiral-wasm/pkg/spiral_wasm_bg.wasm` missing:
      `wasm-pack build --target web --release` in `crates/spiral-wasm/`
    - Symlinks WASM pkg into `ROOT/demo/frontend/pkg`
    - Builds spiral-cpu server if binary missing:
      `cargo build --release -p spiral-cpu-server`
    - Starts spiral server:
      ```bash
      $ROOT/server/target/release/spiral_server \
        --database "$TILES_BIN" \
        --tile-mapping "$TILES_DIR/tile_mapping.json" \
        --num-tiles "$NUM_TILES" \
        --tile-size "$TILE_SIZE" \
        --port 8081
      ```
    - Waits for port 8081 to respond
    - Starts Flask proxy:
      ```bash
      python3 "$ROOT/demo/proxy/server.py" \
        --spiral-port 8081 --port 8002 --tiles-dir "$TILES_DIR"
      ```
    - Prints `http://localhost:8002`
    - `--test-tiles`, `--synthetic`, `--ngrok` flags matching mulpir-gpu
  - **File**: `spiral-cpu/demo/scripts/run_demo.sh` — thin wrapper delegating to `../../run_demo.sh`
    (same pattern as mulpir-gpu/demo/scripts/)
  - **Note**: Default port is 8002 (not 8000) so both demos can run simultaneously for comparison

---

## Implementation Workflow

This plan file is the authoritative checklist. When implementing:

### Required Process
1. **Load Plan**: Read this entire plan file before starting
2. **Sync Tasks**: Create TodoWrite tasks matching the checkboxes above
3. **Execute & Update**: For each task:
   - Mark TodoWrite as `in_progress` when starting
   - Update checkbox `[ ]` to `[x]` when completing
   - Mark TodoWrite as `completed` when done
4. **Maintain Sync**: Keep this file and TodoWrite synchronized throughout

### Critical Rules
- Group A tasks are fully parallel — launch as subagents simultaneously
- Task B1 waits for A1 (shared JS modules) and A2 (spiral-wasm pkg)
- Task C1 waits for A3 (server binary) and B1 (frontend)
- Validate Spiral params with a small integration test before frontend work
- The `mulpir-gpu` refactor (A1) must not change mulpir-gpu behavior — only extract code
- Mark tasks complete only when fully implemented (no placeholders)

### Verification Steps
After all tasks complete:
1. `./mulpir-gpu/run_demo.sh --test-tiles` — original demo still works
2. `./spiral-cpu/run_demo.sh --test-tiles` — new demo works on same tiles
3. Both can run simultaneously (different ports: 8000 and 8002)
4. A code change to `pir-map-shared/frontend/tile-batch.js` is reflected in both demos
