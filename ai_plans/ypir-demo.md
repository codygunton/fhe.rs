# YPIR PIR Map Demo — Implementation Plan

## Executive Summary

Add a YPIR-based PIR map demo following the existing spiral-cpu/mulpir-gpu pattern. The demo will use the YPIR+SP (SimplePIR) mode to serve the same NYC map tile database (~82 MB, 4213 PIR slots × 20480 bytes) via private information retrieval.

**Architecture:**
```
Browser (MapLibre + WASM YPIR client)
    ↕  HTTP on port 8009
Flask Proxy (Python)
    ↕  HTTP on port 8084
YPIR Server (Rust, actix-web, AVX-512)
    reads tiles.bin at startup
```

**Key technical decisions:**
- **YPIR+SP mode** — our tiles (20,480 bytes = 163,840 bits) exceed the 8-bit standard YPIR limit. YPIR+SP treats each item as multiple RLWE plaintext slots (`instances = ceil(163840 / (2048×14)) = 6`).
- **WASM client** — a new `ypir-wasm` crate wrapping the YPIR client-side logic. Since ypir's `lib.rs` requires nightly `#![feature(stdarch_x86_avx512)]`, the WASM crate will depend on `spiral-rs` directly (the local fork) and reimplement the thin YPIR+SP client logic (query generation + response decryption) without depending on the ypir crate. The server-only AVX-512 code stays server-side.
- **AVX-512 enabled** — the server builds with `target-cpu=native` per the YPIR README best practices. The C++ matmul.cpp is compiled with `-march=native -O3`. Requires nightly Rust (`nightly-2024-02-07`).
- **Shared frontend** — reuses `pir-map-shared/frontend/` modules (map-setup.js, tile-batch.js, tile-decoder.js, tile-cache.js) unchanged.
- **Shared tiles** — falls back to `mulpir-gpu/demo/tiles/` for the NYC tile data.

### Data Flow

```
Startup:
  Server loads tiles.bin → transposes into col-major DB
  Server runs offline precomputation: hint_0 = A·DB (seeded), ring-packing precomp
  Server listens on :8084

Per-session:
  Browser → POST /api/setup [expansion keys] → Server stores keys, returns UUID

Per-query batch:
  Browser generates RLWE queries (WASM) for each tile slot
  Browser → POST /api/query-batch [UUID + queries] → Server
  Server: first pass (AVX-512 matmul DB·query), ring packing, modulus switch
  Server → [RLWE ciphertexts] → Browser
  Browser decrypts RLWE ciphertexts (WASM) → raw tile bytes → pako gunzip → PBF
```

### Co-residency with spiral-gpu Agent

Another agent is actively working on `spiral-gpu/` in a separate session. To avoid conflicts:

**Ports they may be using:**
- 8080 (mulpir-gpu backend)
- 8081 (spiral-cpu backend)
- 8082 (spiral-gpu backend)
- 8002 (shared proxy port for all existing demos)

**Our ports (chosen to avoid all collisions):**
- 8084 (YPIR backend server)
- 8009 (YPIR proxy — user-requested)

**Files/directories we MUST NOT touch:**
- `spiral-gpu/` — entirely off-limits, the other agent is actively modifying it
- `spiral-cpu/` — read-only reference; do not modify any files
- `mulpir-gpu/` — read-only; we only read `demo/tiles/` for shared tile data
- `mulpir-cpu/` — read-only reference
- `pir-map-shared/` — read-only; we import from it but do not modify it
- `sdk/lib/spiral-rs/` — read-only dependency; do not modify
- `crates/spiral-wasm/` — belongs to spiral-cpu, do not touch
- `crates/fhe-wasm/` — belongs to mulpir, do not touch

**Files/directories we CREATE (new, no conflicts):**
- `ypir-cpu/` — entirely new directory, our workspace
- `crates/ypir-wasm/` — new WASM crate, does not conflict with existing crates

**Build isolation:**
- Our server has its own `rust-toolchain.toml` (nightly) scoped to `ypir-cpu/server/`
- Our WASM crate uses the workspace default toolchain
- We never run `cargo build` in any existing crate's directory
- We do NOT add ypir-cpu or ypir-wasm to the workspace `Cargo.toml` members list (they are standalone)

**Runtime isolation:**
- `run_demo.sh` only kills processes on OUR ports (8084, 8009) — never 8080/8081/8082/8002
- We never start/stop/restart any existing demo's processes

### Expected Outcomes

- `./ypir-cpu/run_demo.sh` launches the full demo on `localhost:8009`
- The map viewer loads and displays NYC vector tiles fetched via YPIR PIR
- Server uses AVX-512 for the matrix-vector multiply (first pass)
- Client-side WASM handles key generation, query encryption, response decryption

## Goals & Objectives

### Primary Goals
- Working YPIR PIR map demo matching the existing demo UX pattern
- AVX-512 optimizations enabled on the server (per YPIR README)
- WASM client that runs in-browser for true client-side privacy

### Secondary Objectives
- Clean separation: ypir/ stays untouched (it's a symlink to upstream)
- Reuse shared infrastructure (pir-map-shared/, prepare_tiles.py)

## Implementation Tasks

### Visual Dependency Tree

```
ypir-cpu/
├── server/
│   ├── Cargo.toml                    (Task #1: Server crate setup)
│   ├── .cargo/config.toml            (Task #1: target-cpu=native)
│   ├── rust-toolchain.toml           (Task #1: nightly-2024-02-07)
│   └── src/
│       └── main.rs                   (Task #2: HTTP server wrapping YPIR)
│
├── demo/
│   ├── frontend/
│   │   ├── index.html               (Task #4: Frontend files)
│   │   ├── app.js                   (Task #4: YPIR client + batch logic)
│   │   ├── style.css                (Task #4: Copy from spiral-cpu)
│   │   └── pkg/ → symlink           (created by run_demo.sh)
│   ├── proxy/
│   │   └── server.py                (Task #5: Flask proxy)
│   └── tiles/ → fallback            (resolved by run_demo.sh)
│
└── run_demo.sh                       (Task #6: Orchestration script)

crates/
└── ypir-wasm/
    ├── Cargo.toml                    (Task #3: WASM crate setup)
    └── src/
        └── lib.rs                    (Task #3: YPIR+SP client in WASM)
```

### Execution Plan

#### Group A: Foundation (Execute in parallel)

- [ ] **Task #1**: Create YPIR server crate scaffolding
  - Folder: `ypir-cpu/server/`
  - Files: `Cargo.toml`, `.cargo/config.toml`, `rust-toolchain.toml`
  - **`Cargo.toml`**:
    - `name = "ypir-cpu-server"`
    - Dependencies: `ypir = { path = "../../ypir" }`, `spiral-rs = { path = "../../sdk/lib/spiral-rs" }`, `actix-web = "4"`, `tokio`, `clap`, `serde_json`, `tracing`, `tracing-subscriber`, `uuid`, `memmap2`, `anyhow`
    - `[profile.release] opt-level = 3`
  - **`.cargo/config.toml`**:
    ```toml
    [build]
    rustflags = ["-C", "target-cpu=native"]
    ```
  - **`rust-toolchain.toml`**:
    ```toml
    [toolchain]
    channel = "nightly-2024-02-07"
    ```
  - Context: The nightly toolchain is required because `ypir` uses `#![feature(stdarch_x86_avx512)]`. The `target-cpu=native` flag enables AVX-512 codegen for both the YPIR matmul kernels and spiral-rs NTT/poly operations.

- [ ] **Task #3**: Create YPIR WASM client crate
  - Folder: `crates/ypir-wasm/`
  - Files: `Cargo.toml`, `src/lib.rs`
  - **`Cargo.toml`**:
    - `name = "ypir-wasm"`, `crate-type = ["cdylib", "rlib"]`
    - Dependencies: `spiral-rs = { path = "../../sdk/lib/spiral-rs" }`, `wasm-bindgen`, `getrandom = { features = ["js"] }`, `serde_json`, `console_error_panic_hook`, `js-sys`, `rand`, `rand_chacha`
    - NOTE: Does NOT depend on `ypir` crate (avoids nightly AVX-512 requirement for WASM target)
  - **`src/lib.rs`** — `YpirClient` struct exposed via wasm-bindgen:
    - `new(params_json: &str)` — parse YPIR+SP params, create `spiral_rs::client::Client`, leak params as `&'static`
    - `generate_keys() -> Vec<u8>` — generate RLWE secret key + expansion params. Returns serialized expansion key row-1 (the "pub params" the server needs for ring packing). This is the equivalent of `raw_generate_expansion_params()` from YPIR's client.rs, but using spiral-rs directly.
    - `generate_query(target_row: usize) -> Vec<u8>` — generate RLWE query for a DB row. Uses `SEED_0` public randomness (ChaCha20Rng seeded with `[0u8; 32]`). Encrypts a selection vector under the RLWE secret key. Returns the packed query 'b' scalars.
    - `decode_response(data: &[u8]) -> Vec<u8>` — decrypt RLWE ciphertexts from the server response. Recovers plaintext coefficients, extracts the raw tile bytes.
    - `setup_bytes() -> usize`, `query_bytes() -> usize` — size helpers for the frontend
  - **Implementation approach**: The YPIR+SP client logic is thin — it's just spiral-rs `Client` operations (key gen, encrypt, decrypt) with YPIR-specific query formatting (selection vector with `scale_k = modulus / pt_modulus`, negacyclic permutation for ring packing). Extract the logic from `ypir/src/client.rs` lines 193-300 and `ypir/src/scheme.rs` lines 231-258. The core operations are:
    1. Query gen: for each of `2^nu_1` blocks, encrypt a selection polynomial (zero everywhere except at `target_row % poly_len` with value `scale_k`). Use seeded public randomness for the 'a' component.
    2. Response decode: receive `instances` RLWE ciphertexts, each mod-switched. Call `PolyMatrixRaw::recover()` then `decrypt_ct_reg()` from spiral-rs. The plaintext coefficients ARE the tile data (14 bits per coeff, packed).
  - Context: Building this as a separate crate avoids the nightly/AVX-512 dependency. The WASM target compiles spiral-rs in scalar mode (the `#[cfg(target_feature = "avx2")]` guards fall through to scalar paths, which is fine for client-side perf).

#### Group B: Server + WASM Implementation (After Group A)

- [ ] **Task #2**: Implement YPIR HTTP server
  - Folder: `ypir-cpu/server/src/`
  - File: `main.rs`
  - **CLI** (clap):
    - `--database PATH` — path to tiles.bin
    - `--tile-mapping PATH` — path to tile_mapping.json
    - `--num-tiles N` — number of PIR slots
    - `--tile-size N` (default 20480)
    - `--port N` (default 8084)
  - **Startup sequence**:
    1. Parse CLI args
    2. Compute YPIR+SP params via `ypir::params::params_for_scenario_simplepir(num_tiles, item_size_bits)` where `item_size_bits = tile_size * 8`
    3. Load tiles.bin into memory, create `YServer::<u16>::new(&params, tile_data_iter, is_simplepir=true, false, true)`
    4. Run offline precomputation: `y_server.perform_offline_precomputation_simplepir(None)`
    5. Store server + offline values in shared state
    6. Start actix-web HTTP server
  - **Lifetime management**: `Params` is `Box::leak`'d to `&'static`. `YServer` and `OfflinePrecomputedValues` both borrow `'static` params. Wrap in `Arc<ServerState>`. Since `perform_online_computation_simplepir` takes `&self` (not `&mut self`) for the SimplePIR path, concurrent queries should work without cloning. Verify this — if mutation occurs, use `Mutex<OfflinePrecomputedValues>` and clone before each query.
  - **Endpoints**:
    - `GET /api/params` → JSON: `{ num_tiles, tile_size, ypir_params: <spiral params JSON>, setup_bytes, query_bytes, response_bytes, num_items, instances }`
    - `GET /api/tile-mapping` → tile_mapping.json contents
    - `POST /api/setup` → body = serialized expansion key bytes. Store in `HashMap<UUID, Vec<u8>>`. Return UUID string.
    - `POST /api/query-batch` → body = `[UUID:36][count:u32LE][q0_bytes]...[qN_bytes]`. For each query:
      1. Deserialize expansion key pub params from stored session
      2. Deserialize packed query row from request bytes
      3. Call `y_server.perform_online_computation_simplepir(packed_query, &offline_vals, &[&pub_params], None)`
      4. Serialize response RLWE ciphertexts as contiguous bytes
      5. Return concatenated responses
  - **DB loading**: Read tiles.bin as raw bytes. Convert to the iterator of `u16` values that `YServer::new` expects. The tile data (bytes) must be mapped into the RLWE plaintext space: each byte becomes a `u16` mod `pt_modulus` (which is `1 << 14 = 16384` for YPIR+SP). The mapping is: lay out tile bytes as polynomial coefficients, with 14 bits per coefficient packing.
    - Actually, `YServer::new` takes a `pt_iter: impl Iterator<Item = T>` and internally packs data into the DB matrix of shape `db_rows × db_cols`. For tile data: feed the raw tile bytes mapped to u16 values, and the server handles the layout. Study how the benchmark in `scheme.rs` does it (line 97: `let pt_iter = std::iter::repeat_with(|| (T::sample() as u64 % params.pt_modulus) as T)`). For real tile data, read bytes from tiles.bin and feed them as the plaintext iterator.
    - **Critical detail**: The DB layout for YPIR+SP is `db_rows × db_cols` where each row is one item (tile). `db_rows = 2^(nu_1 + poly_len_log2)` and `db_cols = instances × poly_len`. Each element is a u16 mod pt_modulus. The raw tile bytes need to be packed into 14-bit coefficients across `instances` RLWE plaintexts per row.
  - Context: This is the most complex task. The YPIR library API is designed for benchmarking, not serving — the main challenge is correctly loading real tile data into the DB format and handling the lifetime/ownership issues.

#### Group C: Frontend + Proxy (Can start in parallel with Group B for proxy; frontend needs WASM from Task #3)

- [ ] **Task #4**: Create YPIR demo frontend
  - Folder: `ypir-cpu/demo/frontend/`
  - Files: `index.html`, `app.js`, `style.css`
  - **`index.html`**: Copy from `spiral-cpu/demo/frontend/index.html`, change title to "PIR Map Demo — YPIR CPU", subtitle to "Map tiles fetched via YPIR PIR (CPU)"
  - **`style.css`**: Copy from `spiral-cpu/demo/frontend/style.css` unchanged
  - **`app.js`**: Follow the pattern of `spiral-cpu/demo/frontend/app.js` exactly:
    1. Import `YpirClient` from `./pkg/ypir_wasm.js` (the WASM module)
    2. Import shared modules from `/shared/` (tile-cache, tile-batch, tile-decoder, map-setup)
    3. `initialize()`:
       - Load WASM, fetch `/api/params`, create `YpirClient(params.ypir_params)`
       - Generate keys via `client.generate_keys()`, POST to `/api/setup`, get UUID
       - Fetch `/api/tile-mapping`, init map
    4. `ypirBackend.processBatch(tiles, abortSignal)`:
       - For each tile's slots, generate queries via `client.generate_query(pirIdx)`
       - Build batch payload: `[UUID:36][count:u32LE][q0][q1]...`
       - POST to `/api/query-batch`
       - Slice response into per-query chunks (`response_bytes` each)
       - Decrypt each: `client.decode_response(chunk)` → raw slot bytes
       - Decode via `decodeSlotToPBF` / `decodeMultiSlotToPBF`
    5. Same metrics polling, loading screen, PIR badge as spiral-cpu
  - Context: The frontend is a near-copy of the spiral-cpu frontend with the WASM client swapped out. The batch API contract (UUID + count + queries → concatenated responses) is identical in structure.

- [ ] **Task #5**: Create YPIR Flask proxy
  - Folder: `ypir-cpu/demo/proxy/`
  - File: `server.py`
  - Copy from `spiral-cpu/demo/proxy/server.py` with these changes:
    - Rename `--spiral-host/--spiral-port` to `--ypir-host/--ypir-port`, default port 8084
    - Forward endpoints: `/api/setup`, `/api/query-batch` (replacing `/api/private-read-batch`), `/api/params`, `/api/tile-mapping`
    - Remove `/api/private-read` (single query) — YPIR will only support batch
    - Keep `/api/metrics`, shared file serving, static file serving unchanged
    - Import `common.py` from `pir-map-shared/proxy/`
  - Context: This is a thin HTTP-to-HTTP forwarder, identical in structure to the spiral-cpu proxy. The only differences are endpoint names and backend port.

#### Group D: Orchestration (After Groups B and C)

- [ ] **Task #6**: Create run_demo.sh
  - File: `ypir-cpu/run_demo.sh`
  - Follow `spiral-cpu/run_demo.sh` exactly with these substitutions:
    - WASM crate: `crates/ypir-wasm` (build with `wasm-pack build --target web --release`)
    - Server binary: `ypir-cpu/server/target/release/ypir-cpu-server`
    - Server port: 8084
    - Proxy port: 8009 (user requested `localhost:8009`)
    - Proxy script: `ypir-cpu/demo/proxy/server.py --ypir-port 8084 --port 8009`
    - Tile data: look for `ypir-cpu/demo/tiles/`, fall back to `mulpir-gpu/demo/tiles/`
    - Build command: `cd ypir-cpu/server && cargo build --release` (nightly toolchain is picked up from rust-toolchain.toml)
  - **PORT SAFETY**: The `kill_port` helper MUST only target ports 8084 and 8009. NEVER kill processes on 8080, 8081, 8082, or 8002 — another agent may be running spiral-gpu services on those ports.
  - Full script pattern:
    1. Build ypir-wasm WASM if `pkg/ypir_wasm_bg.wasm` missing
    2. Symlink WASM pkg into `demo/frontend/pkg/`
    3. Build YPIR server if binary missing
    4. Resolve tiles (local → mulpir-gpu fallback)
    5. Start YPIR server on :8084
    6. Wait for server ready (TCP probe)
    7. Start Flask proxy on :8009
    8. Print URL, wait, cleanup trap
  - Make executable: `chmod +x`

## Known Risks & Mitigations

1. **YPIR library not designed as a server library**: The `YServer`/`OfflinePrecomputedValues` types use borrowed lifetimes. Mitigation: `Box::leak` the params, verify that `perform_online_computation_simplepir` doesn't mutate shared state (it takes `&self` + `&OfflinePrecomputedValues` — looks safe for SimplePIR mode).

2. **Tile data loading**: YPIR expects plaintext data as an iterator of `u16` mod `pt_modulus`. Need to correctly pack raw tile bytes into 14-bit RLWE coefficients. If the packing is wrong, decrypted tiles will be garbage. Mitigation: start with a small test (1-2 tiles), verify round-trip correctness before scaling.

3. **WASM client complexity**: Reimplementing the YPIR+SP client query/decode logic outside the ypir crate is non-trivial. The query generation involves seeded RLWE encryption with specific formatting. Mitigation: port the logic line-by-line from `ypir/src/client.rs` and `ypir/src/scheme.rs`, using the same constants (SEED_0, scale_k, etc.).

4. **Nightly Rust for server only**: The server must use nightly-2024-02-07. The WASM crate uses stable. Separate `rust-toolchain.toml` files handle this (server dir has nightly, crates/ uses workspace default).

5. **Co-resident spiral-gpu agent**: Another agent is actively working on `spiral-gpu/` and may have services running on ports 8080-8082 and 8002. Mitigation: our ports (8084, 8009) are well clear. We create only new directories (`ypir-cpu/`, `crates/ypir-wasm/`). We never modify, build, or kill processes in any existing directory. The `run_demo.sh` kill_port helper ONLY targets our ports. We do NOT add ourselves to the workspace Cargo.toml to avoid triggering workspace-wide rebuilds that could interfere with the other agent's compilation.

## Implementation Workflow

This plan file serves as the authoritative checklist for implementation. When implementing:

### Required Process
1. **Load Plan**: Read this entire plan file before starting
2. **Sync Tasks**: Create TodoWrite tasks matching the checkboxes above
3. **Execute & Update**: For each task:
   - Mark TodoWrite as `in_progress` when starting
   - Update checkbox `[ ]` to `[x]` when completing
   - Mark TodoWrite as `completed` when done
4. **Maintain Sync**: Keep this file and TodoWrite synchronized throughout

### Critical Rules
- This plan file is the source of truth for progress
- Update checkboxes in real-time as work progresses
- Never lose synchronization between plan file and TodoWrite
- Mark tasks complete only when fully implemented (no placeholders)
- Tasks should be run in parallel, unless there are dependencies, using subtasks

### Progress Tracking
The checkboxes above represent the authoritative status of each task. Keep them updated as you work.
