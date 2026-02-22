# mulpir-cpu Demo Implementation Plan

## Executive Summary

> The project has two existing PIR demos: **mulpir-gpu** (BFV/MulPIR on GPU via HEonGPU) and
> **spiral-cpu** (Spiral PIR on CPU via spiral-rs). This plan adds a third demo — **mulpir-cpu**
> — running the same BFV/MulPIR scheme as the GPU demo but entirely on CPU, using the `fhe`
> Rust crate for all cryptographic operations.
>
> The new demo reuses the shared frontend infrastructure (`pir-map-shared/`), follows the
> exact directory and launch-script pattern of `spiral-cpu/`, and exposes an HTTP API that
> mirrors `spiral-cpu`'s UUID-session pattern. A new WASM crate (`crates/mulpir-cpu-wasm/`)
> provides the browser-side client using **fhe.rs native protobuf serialisation** (not HEonGPU
> binary), which keeps the server completely independent of HEonGPU.

## Goals & Objectives

### Primary Goals
- CPU-only MulPIR map demo runnable on any Linux machine without a GPU, producing the same
  privacy guarantee as the GPU demo.
- End-to-end round-trip: tile queried via BFV MulPIR, decrypted in WASM, rendered by MapLibre
  as a vector-tile layer.

### Secondary Objectives
- Demonstrate the performance difference between CPU and GPU for the same PIR scheme.
- Keep the new code minimal by reusing `pir-map-shared/` and `prepare_tiles.py` unchanged.

## Solution Overview

### Approach
The browser generates BFV keys and queries using a new WASM module (`mulpir-cpu-wasm`) that
serialises everything with fhe.rs's native protobuf format. The Rust HTTP server deserialises
these natively, loads the tile database as BFV plaintexts, and executes the MulPIR query
(expand → dot-product column sums → ct×ct accumulate → relinearise → mod-switch). A thin
Flask proxy serves static files and forwards API calls, identical in structure to `spiral-cpu`.

### Key Components
1. **`crates/mulpir-cpu-wasm/`** – New WASM crate. Wraps `fhe` crate: key generation,
   query encryption, response decryption. Uses `EvaluationKey.to_bytes()` / `from_bytes()`.
2. **`mulpir-cpu/server/`** – Standalone Rust workspace. Actix-web HTTP server that loads
   the tile DB as `Vec<Plaintext>`, manages sessions (UUID → keys), and processes MulPIR
   queries entirely in software.
3. **`mulpir-cpu/demo/frontend/`** – Browser app, nearly identical to `spiral-cpu/demo/frontend/`
   but uses `PIRClient` from `mulpir-cpu-wasm` and calls `/api/batch-query`.
4. **`mulpir-cpu/demo/proxy/`** – Flask proxy (copy of `spiral-cpu/demo/proxy/server.py`
   with adjusted endpoint forwarding).
5. **`mulpir-cpu/run_demo.sh`** – Launch script matching `spiral-cpu/run_demo.sh`.

### Architecture Diagram

```
Browser
  mulpir-cpu-wasm (PIRClient)
      │  generate_galois_key() → protobuf bytes
      │  generate_relin_key()  → protobuf bytes
      │  create_query(idx)     → protobuf Ciphertext bytes
      │  decrypt_response()    → tile bytes
      │
      └─ HTTP ─► Flask proxy (port 8002)
                    │  /api/setup         ─► Rust server (port 8081)
                    │  /api/batch-query   ─►    load_db (Vec<Plaintext>)
                    │  /api/params        ─►    expand → dot_product_scalar → ct*ct
                    │  /api/tile-mapping  ─►    relinearise → mod-switch
                    │  /shared/*          ─► pir-map-shared/frontend/
                    └─ static files      ─► mulpir-cpu/demo/frontend/
```

### Data Flow

```
Setup:
  Browser ──[galois_key_len:4B][galois_bytes][relin_bytes]──► POST /api/setup ──► UUID

Query (per batch):
  Browser ──[UUID:36B][num_queries:4B][q_len:4B][q_bytes]...──► POST /api/batch-query
  Server:  expand query ──► col dot-products ──► ct×ct accumulate ──► relin ──► mod-switch
  Server ──[num:4B][r_len:4B][r_bytes]...──► Browser
  Browser:  decrypt_response() ──► tile bytes ──► decodeSlotToPBF ──► MapLibre
```

### Expected Outcomes
- `./run_demo.sh --synthetic` produces a working interactive PIR map demo on CPU.
- Tiles render as landcover fills (via synthetic MVT), confirming the full pipeline.
- CPU query latency is measurably higher than the GPU demo (~seconds vs. ~100 ms).
- The demo can also be run with real `.mbtiles` data via `prepare_tiles.py`.

---

## Implementation Tasks

### CRITICAL IMPLEMENTATION RULES
1. **NO PLACEHOLDER CODE** – every function must be fully implemented.
2. **fhe.rs native serialisation only** – no HEonGPU binary format in mulpir-cpu-wasm or server.
3. **Standalone server workspace** – `mulpir-cpu/server/Cargo.toml` must have `[workspace]`
   to prevent the root workspace from pulling in actix-web.
4. **WASM in root workspace** – `crates/mulpir-cpu-wasm/` is picked up automatically by
   `members = ["crates/*"]`; use `workspace = true` for shared fields.
5. **Session management** – UUIDs identify (EvaluationKey, RelinearizationKey) pairs stored
   server-side in a `RwLock<HashMap<String, SessionKeys>>`.

### BFV Parameters (hardcoded, same as `fhe-wasm`)

```rust
const DEGREE: usize = 8192;
const PLAINTEXT_MODULUS: u64 = (1<<20)|(1<<19)|(1<<17)|(1<<16)|(1<<14)|1; // 1_785_857
const MODULI_SIZES: [usize; 3] = [50, 55, 55]; // Q primes
const BITS_PER_COEFF: usize = 20;
const BYTES_PER_PLAINTEXT: usize = BITS_PER_COEFF * DEGREE / 8; // 20 480
```

### Dimension Computation (shared logic, used in both WASM and server)

```rust
fn compute_dims(num_tiles: usize, tile_size: usize) -> (usize, usize, usize, usize) {
    // (dim1, dim2, expansion_level, elements_per_plaintext)
    let elements_per_pt = (BYTES_PER_PLAINTEXT / tile_size).max(1);
    let num_rows = num_tiles.div_ceil(elements_per_pt);
    let dim1 = (num_rows as f64).sqrt().ceil() as usize;
    let dim2 = num_rows.div_ceil(dim1);
    let expansion_level = (dim1 + dim2).next_power_of_two().trailing_zeros() as usize;
    (dim1, dim2, expansion_level, elements_per_pt)
}
```

### Galois Elements for Expansion (used in WASM keygen)

```rust
// For each l in 0..expansion_level: galois_element = (DEGREE >> l) + 1
let galois_elements: Vec<u64> = (0..expansion_level)
    .map(|l| ((DEGREE >> l) + 1) as u64)
    .collect();
```

### MulPIR Server Query Processing Algorithm

```rust
// All types from the `fhe` crate: Ciphertext, EvaluationKey, RelinearizationKey, Plaintext
fn process_query(
    params: &Arc<BfvParameters>,
    ek: &EvaluationKey,
    rk: &RelinearizationKey,
    db: &[Plaintext],        // dim1 * dim2 plaintexts in row-major order
    dim1: usize,
    dim2: usize,
    query_bytes: &[u8],
) -> Result<Vec<u8>> {
    use fhe::bfv;
    let query = Ciphertext::from_bytes(query_bytes, params)?;
    let expanded = ek.expands(&query, dim1 + dim2)?;   // Vec<Ciphertext> len = dim1+dim2

    let mut result = bfv::Ciphertext::zero(params);
    for j in 0..dim2 {
        // Column j: db entries at indices j, j+dim2, j+2*dim2, ..., j+(dim1-1)*dim2
        let col_ct = bfv::dot_product_scalar(
            expanded[..dim1].iter(),
            db.iter().skip(j).step_by(dim2),
        )?;
        result += &(&col_ct * &expanded[dim1 + j]);
    }

    rk.relinearizes(&mut result)?;
    result.switch_to_level(result.max_switchable_level())?;
    Ok(result.to_bytes())
}
```

### DB Loading Algorithm (server startup)

```rust
// tiles.bin from prepare_tiles.py: flat concatenation, tile_i at offset i*tile_size
// Each tile (tile_size bytes) packed into a degree-8192 polynomial at BITS_PER_COEFF bits/coeff
fn load_db(params: &Arc<BfvParameters>, raw: &[u8], num_tiles: usize, tile_size: usize)
    -> Vec<Plaintext>
{
    let (dim1, dim2, _, elements_per_pt) = compute_dims(num_tiles, tile_size);
    let total_slots = dim1 * dim2;
    let coeffs_per_elem = tile_size * 8 / BITS_PER_COEFF; // 8192 for tile_size=20480

    (0..total_slots).map(|slot| {
        let mut coeff_values = vec![0u64; DEGREE];
        for elem in 0..elements_per_pt {
            let tile_idx = slot * elements_per_pt + elem;
            if tile_idx < num_tiles {
                let tile_data = &raw[tile_idx * tile_size .. (tile_idx+1) * tile_size];
                let base = elem * coeffs_per_elem;
                for i in 0..coeffs_per_elem {
                    let bit_start = i * BITS_PER_COEFF;
                    let mut val = 0u64;
                    for b in 0..BITS_PER_COEFF {
                        let total_bit = bit_start + b;
                        let byte_idx = total_bit / 8;
                        let bit_idx = total_bit % 8;
                        if byte_idx < tile_data.len() {
                            val |= (((tile_data[byte_idx] >> bit_idx) & 1) as u64) << b;
                        }
                    }
                    coeff_values[base + i] = val;
                }
            }
        }
        Plaintext::try_encode(&coeff_values, Encoding::poly(), params).unwrap()
    }).collect()
}
```

### Query Wire Format (WASM → server)

Selection vector encoding (creates encrypted query for tile `tile_index`):
```rust
// Requires: inv = mod_inverse(1 << expansion_level, plaintext_modulus)
let slot = tile_index / elements_per_pt;
let row  = slot / dim2;
let col  = slot % dim2;
let mut pt_vec = vec![0u64; dim1 + dim2];
pt_vec[row]        = inv;
pt_vec[dim1 + col] = inv;
let pt = Plaintext::try_encode(&pt_vec, Encoding::poly_at_level(1), &params)?;
let ct = sk.try_encrypt(&pt, &mut rng)?;
// Serialise: ct.to_bytes()  (fhe.rs protobuf)
```

Response decryption (WASM):
```rust
let ct = Ciphertext::from_bytes(response_bytes, &params)?;
let pt = sk.try_decrypt(&ct)?;
let coeffs = pt.poly_iter().next().unwrap();  // or appropriate accessor
let elem = tile_index % elements_per_pt;
let base = elem * coeffs_per_elem;
// Unpack BITS_PER_COEFF-bit coefficients back to bytes
let tile_bytes: Vec<u8> = (0 .. tile_size).map(|byte_idx| {
    let bit_start = byte_idx * 8;
    let mut byte = 0u8;
    for b in 0..8usize {
        let total_bit = bit_start + b;
        let coeff_idx = base + total_bit / BITS_PER_COEFF;
        let coeff_bit = total_bit % BITS_PER_COEFF;
        let bit = ((coeffs[coeff_idx] >> coeff_bit) & 1) as u8;
        byte |= bit << b;
    }
    byte
}).collect();
```

> **Note on Plaintext accessor**: Check how `fhe-wasm/src/lib.rs`'s `decrypt_response`
> accesses polynomial coefficients — it likely calls `pt.coefficients()` or similar. Mirror
> that exact accessor in `mulpir-cpu-wasm/src/lib.rs`.

---

### Visual Dependency Tree

```
fhe.rs/
├── crates/
│   └── mulpir-cpu-wasm/          (Task A1 — new WASM crate)
│       ├── Cargo.toml
│       └── src/lib.rs
│
├── mulpir-cpu/
│   ├── server/                   (Task A2 — new standalone Rust workspace)
│   │   ├── Cargo.toml            ([workspace] table + dependencies)
│   │   └── src/main.rs           (actix-web server, DB loading, query processing)
│   │
│   ├── demo/
│   │   ├── frontend/             (Task B1)
│   │   │   ├── index.html        (CPU metrics panel, "MulPIR CPU" branding)
│   │   │   └── app.js            (imports mulpir_cpu_wasm.js, PIRClient, batch-query flow)
│   │   │
│   │   └── proxy/                (Task B2)
│   │       └── server.py         (Flask: /api/setup, /api/batch-query, /shared/, static)
│   │
│   └── run_demo.sh               (Task B3)
│
└── .gitignore                    (Task B3 — add mulpir-cpu/server/target/ and demo/frontend/pkg)
```

---

### Execution Plan

#### Group A: Core Implementation (Execute A1 and A2 in parallel)

- [x] **Task A1**: Create `crates/mulpir-cpu-wasm/`
  - **Folder**: `crates/mulpir-cpu-wasm/`
  - **Files**: `Cargo.toml`, `src/lib.rs`

  **`Cargo.toml`**:
  ```toml
  [package]
  name = "mulpir-cpu-wasm"
  version = "0.1.0"
  edition.workspace = true
  rust-version.workspace = true

  [lib]
  crate-type = ["cdylib", "rlib"]

  [dependencies]
  fhe = { path = "../fhe" }
  fhe-traits = { path = "../fhe-traits" }
  fhe-util = { path = "../fhe-util" }
  rand.workspace = true
  prost.workspace = true
  wasm-bindgen = "0.2"
  js-sys = "0.3"
  getrandom = { version = "0.3", features = ["wasm_js"] }
  console_error_panic_hook = "0.1"
  serde_json = "1"
  ```

  **`src/lib.rs`** — implement ALL of:

  ```rust
  use wasm_bindgen::prelude::*;
  use fhe::bfv::{BfvParametersBuilder, SecretKey, Plaintext, Ciphertext, Encoding,
                  EvaluationKeyBuilder, RelinearizationKey};
  use fhe_traits::{Serialize, Deserialize, DeserializeParametrized};
  use rand::thread_rng;
  use std::sync::Arc;

  // BFV constants
  const DEGREE: usize = 8192;
  const PLAINTEXT_MODULUS: u64 = (1<<20)|(1<<19)|(1<<17)|(1<<16)|(1<<14)|1;
  const MODULI_SIZES: [usize; 3] = [50, 55, 55];
  const BITS_PER_COEFF: usize = 20;
  const BYTES_PER_PLAINTEXT: usize = BITS_PER_COEFF * DEGREE / 8; // 20480

  fn build_params() -> Arc<fhe::bfv::BfvParameters> {
      BfvParametersBuilder::new()
          .set_degree(DEGREE)
          .set_plaintext_modulus(PLAINTEXT_MODULUS)
          .set_moduli_sizes(&MODULI_SIZES)
          .build_arc()
          .expect("BFV params")
  }

  fn compute_dims(num_tiles: usize, tile_size: usize) -> (usize, usize, usize, usize) {
      let elements_per_pt = (BYTES_PER_PLAINTEXT / tile_size).max(1);
      let num_rows = num_tiles.div_ceil(elements_per_pt);
      let dim1 = (num_rows as f64).sqrt().ceil() as usize;
      let dim2 = num_rows.div_ceil(dim1);
      let expansion_level = (dim1 + dim2).next_power_of_two().trailing_zeros() as usize;
      (dim1, dim2, expansion_level, elements_per_pt)
  }

  fn mod_inverse(a: u64, m: u64) -> u64 {
      // Extended Euclidean algorithm
      // Returns x such that a*x ≡ 1 (mod m)
      // Implement using i128 to avoid overflow
  }

  #[wasm_bindgen]
  pub struct PIRClient {
      params: Arc<fhe::bfv::BfvParameters>,
      sk: SecretKey,
      ek: fhe::bfv::EvaluationKey,
      rk: RelinearizationKey,
      dim1: usize,
      dim2: usize,
      expansion_level: usize,
      elements_per_pt: usize,
      tile_size: usize,
  }

  #[wasm_bindgen]
  impl PIRClient {
      /// Construct a new PIR client for `num_tiles` tiles of `tile_size` bytes.
      /// Generates a fresh secret key, evaluation key (Galois keys), and relin key.
      pub fn new(num_tiles: usize, tile_size: usize) -> Result<PIRClient, JsError> {
          console_error_panic_hook::set_once();
          let params = build_params();
          let mut rng = thread_rng();
          let (dim1, dim2, expansion_level, elements_per_pt) = compute_dims(num_tiles, tile_size);

          let sk = SecretKey::random(&params, &mut rng);

          // Galois keys for expansion: one key per Galois element g = (DEGREE >> l) + 1
          let galois_elements: Vec<u64> = (0..expansion_level)
              .map(|l| ((DEGREE >> l) + 1) as u64)
              .collect();
          let ek = EvaluationKeyBuilder::new_leveled(&sk, 1, 0)?
              // enable_galois_key for each element
              // (use the API in EvaluationKeyBuilder to add all Galois elements)
              // Mirror how fhe-wasm/src/lib.rs builds its EvaluationKey
              .build(&mut rng)?;

          let rk = RelinearizationKey::new_leveled(&sk, 0, 0, &mut rng)?;

          Ok(PIRClient { params, sk, ek, rk, dim1, dim2, expansion_level, elements_per_pt, tile_size })
      }

      /// Serialise the Galois evaluation key (fhe.rs protobuf format).
      pub fn generate_galois_key(&self) -> Result<Vec<u8>, JsError> {
          Ok(self.ek.to_bytes())
      }

      /// Serialise the relinearisation key (fhe.rs protobuf format).
      pub fn generate_relin_key(&self) -> Result<Vec<u8>, JsError> {
          Ok(self.rk.to_bytes())
      }

      /// Create an encrypted query for the given tile index.
      pub fn create_query(&self, tile_index: usize) -> Result<Vec<u8>, JsError> {
          let mut rng = thread_rng();
          let inv = mod_inverse(1u64 << self.expansion_level, PLAINTEXT_MODULUS);
          let slot = tile_index / self.elements_per_pt;
          let row  = slot / self.dim2;
          let col  = slot % self.dim2;
          let mut pt_vec = vec![0u64; self.dim1 + self.dim2];
          pt_vec[row]             = inv;
          pt_vec[self.dim1 + col] = inv;
          let pt = Plaintext::try_encode(&pt_vec, Encoding::poly_at_level(1), &self.params)?;
          let ct = self.sk.try_encrypt(&pt, &mut rng)?;
          Ok(ct.to_bytes())
      }

      /// Decrypt a server response and return the tile bytes.
      pub fn decrypt_response(&self, response_bytes: &[u8], tile_index: usize)
          -> Result<Vec<u8>, JsError>
      {
          let ct = Ciphertext::from_bytes(response_bytes, &self.params)?;
          let pt = self.sk.try_decrypt(&ct)?;
          // Extract polynomial coefficients (refer to fhe-wasm/src/lib.rs for exact accessor)
          let coeffs: &[u64] = /* pt accessor */;
          let coeffs_per_elem = self.tile_size * 8 / BITS_PER_COEFF;
          let elem = tile_index % self.elements_per_pt;
          let base = elem * coeffs_per_elem;
          let tile_bytes: Vec<u8> = (0..self.tile_size).map(|byte_idx| {
              let bit_start = byte_idx * 8;
              let mut byte = 0u8;
              for b in 0..8usize {
                  let total_bit = bit_start + b;
                  let coeff_idx = base + total_bit / BITS_PER_COEFF;
                  let coeff_bit = total_bit % BITS_PER_COEFF;
                  let bit = ((coeffs[coeff_idx] >> coeff_bit) & 1) as u8;
                  byte |= bit << b;
              }
              byte
          }).collect();
          Ok(tile_bytes)
      }

      /// JSON string: {"num_tiles":N,"tile_size":N,"dim1":N,"dim2":N,"expansion_level":N,...}
      pub fn get_params_json(&self) -> String {
          serde_json::json!({
              "dim1": self.dim1,
              "dim2": self.dim2,
              "expansion_level": self.expansion_level,
              "elements_per_plaintext": self.elements_per_pt,
              "tile_size": self.tile_size,
          }).to_string()
      }

      pub fn expansion_level(&self) -> usize { self.expansion_level }
  }
  ```

  > **Critical implementation note on `EvaluationKeyBuilder`**: Check `fhe-wasm/src/lib.rs`
  > lines generating Galois keys to find the exact builder call chain. The builder likely
  > needs `.enable_galois_key(galois_element)?` for each element before `.build()`.
  > Mirror that pattern exactly.
  >
  > **Critical note on Plaintext coefficient accessor**: In `decrypt_response`, find the
  > correct accessor for polynomial coefficients from a decrypted `Plaintext` by checking
  > how `fhe-wasm/src/lib.rs` does it and using the same accessor.

---

- [x] **Task A2**: Create `mulpir-cpu/server/`
  - **Folder**: `mulpir-cpu/server/`
  - **Files**: `Cargo.toml`, `src/main.rs`

  **`Cargo.toml`**:
  ```toml
  [workspace]

  [package]
  name = "mulpir-cpu-server"
  version = "0.1.0"
  edition = "2021"
  rust-version = "1.88"

  [[bin]]
  name = "mulpir_cpu_server"
  path = "src/main.rs"

  [dependencies]
  fhe = { path = "../../crates/fhe", features = [] }
  fhe-traits = { path = "../../crates/fhe-traits" }
  actix-web = { version = "4", features = ["macros"] }
  tokio = { version = "1", features = ["full"] }
  clap = { version = "4", features = ["derive"] }
  serde = { version = "1", features = ["derive"] }
  serde_json = "1"
  uuid = { version = "1", features = ["v4"] }
  memmap2 = "0.9"
  anyhow = "1"
  tracing = "0.1"
  tracing-subscriber = { version = "0.3", features = ["env-filter"] }
  rand = "0.9"

  [profile.release]
  opt-level = 3
  ```

  **`src/main.rs`** — implement ALL of:

  ```rust
  use std::collections::HashMap;
  use std::io::Cursor;
  use std::sync::{Arc, RwLock};
  use actix_web::{middleware, web, App, HttpResponse, HttpServer};
  use anyhow::Context;
  use clap::Parser;
  use fhe::bfv::{BfvParametersBuilder, Plaintext, Encoding, EvaluationKey,
                  RelinearizationKey, Ciphertext, BfvParameters};
  use fhe_traits::{Serialize, Deserialize, DeserializeParametrized};
  use memmap2::Mmap;
  use tracing::{error, info};
  use uuid::Uuid;

  // ── Constants ─────────────────────────────────────────────────────────────
  const DEGREE: usize = 8192;
  const PLAINTEXT_MODULUS: u64 = (1<<20)|(1<<19)|(1<<17)|(1<<16)|(1<<14)|1;
  const MODULI_SIZES: [usize; 3] = [50, 55, 55];
  const BITS_PER_COEFF: usize = 20;
  const BYTES_PER_PLAINTEXT: usize = BITS_PER_COEFF * DEGREE / 8; // 20 480

  // ── CLI ───────────────────────────────────────────────────────────────────
  #[derive(Parser)]
  #[command(name = "mulpir-cpu-server")]
  struct Cli {
      #[arg(long)] database: String,
      #[arg(long)] tile_mapping: String,
      #[arg(long)] num_tiles: usize,
      #[arg(long, default_value_t = 20480)] tile_size: usize,
      #[arg(long, default_value_t = 8081)] port: u16,
  }

  // ── Session ───────────────────────────────────────────────────────────────
  struct SessionKeys {
      ek: EvaluationKey,
      rk: RelinearizationKey,
  }

  // ── Server state ──────────────────────────────────────────────────────────
  struct ServerState {
      params: Arc<BfvParameters>,
      db: Vec<Plaintext>,            // dim1 * dim2 plaintexts, row-major
      dim1: usize,
      dim2: usize,
      expansion_level: usize,
      elements_per_pt: usize,
      num_tiles: usize,
      tile_size: usize,
      tile_mapping_json: String,
      sessions: RwLock<HashMap<String, SessionKeys>>,
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  fn compute_dims(num_tiles: usize, tile_size: usize) -> (usize, usize, usize, usize) {
      let elements_per_pt = (BYTES_PER_PLAINTEXT / tile_size).max(1);
      let num_rows = num_tiles.div_ceil(elements_per_pt);
      let dim1 = (num_rows as f64).sqrt().ceil() as usize;
      let dim2 = num_rows.div_ceil(dim1);
      let expansion_level = (dim1 + dim2).next_power_of_two().trailing_zeros() as usize;
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
      let coeffs_per_elem = tile_size * 8 / BITS_PER_COEFF;

      (0..total_slots).map(|slot| {
          let mut coeff_values = vec![0u64; DEGREE];
          for elem in 0..elements_per_pt {
              let tile_idx = slot * elements_per_pt + elem;
              if tile_idx < num_tiles {
                  let tile_data = &raw[tile_idx * tile_size..(tile_idx + 1) * tile_size];
                  let base = elem * coeffs_per_elem;
                  for i in 0..coeffs_per_elem {
                      let bit_start = i * BITS_PER_COEFF;
                      let mut val = 0u64;
                      for b in 0..BITS_PER_COEFF {
                          let total_bit = bit_start + b;
                          let byte_idx = total_bit / 8;
                          let bit_idx = total_bit % 8;
                          if byte_idx < tile_data.len() {
                              val |= (((tile_data[byte_idx] >> bit_idx) & 1) as u64) << b;
                          }
                      }
                      coeff_values[base + i] = val;
                  }
              }
          }
          Plaintext::try_encode(&coeff_values, Encoding::poly(), params).expect("encode")
      }).collect()
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
      use fhe::bfv;
      let query = Ciphertext::from_bytes(query_bytes, params)?;
      let expanded = ek.expands(&query, dim1 + dim2)?;

      let mut result = bfv::Ciphertext::zero(params);
      for j in 0..dim2 {
          let col_ct = bfv::dot_product_scalar(
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
      let galois_len = u32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
      if body.len() < 4 + galois_len {
          return HttpResponse::BadRequest().body("body too short for galois key");
      }
      let galois_bytes = &body[4..4 + galois_len];
      let relin_bytes  = &body[4 + galois_len..];

      let params = state.params.clone();
      let galois_bytes = galois_bytes.to_vec();
      let relin_bytes  = relin_bytes.to_vec();

      let result = web::block(move || {
          let ek = EvaluationKey::from_bytes(&galois_bytes, &params)?;
          let rk = RelinearizationKey::from_bytes(&relin_bytes, &params)?;
          Ok::<_, anyhow::Error>((ek, rk))
      }).await;

      match result {
          Ok(Ok((ek, rk))) => {
              let uuid = Uuid::new_v4().to_string();
              state.sessions.write().unwrap()
                  .insert(uuid.clone(), SessionKeys { ek, rk });
              info!(uuid = %uuid, "stored session keys");
              HttpResponse::Ok().content_type("text/plain").body(uuid)
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

      // Parse individual queries from the body
      let mut queries: Vec<Vec<u8>> = Vec::new();
      let num_queries = u32::from_le_bytes(body[UUID_LEN..UUID_LEN+4].try_into().unwrap()) as usize;
      let mut off = UUID_LEN + 4;
      for _ in 0..num_queries {
          if off + 4 > body.len() { break; }
          let q_len = u32::from_le_bytes(body[off..off+4].try_into().unwrap()) as usize;
          off += 4;
          if off + q_len > body.len() { break; }
          queries.push(body[off..off+q_len].to_vec());
          off += q_len;
      }

      // Fetch session keys
      let (ek_bytes, rk_bytes) = {
          let sessions = state.sessions.read().unwrap();
          match sessions.get(&uuid) {
              None => return HttpResponse::NotFound()
                  .body(format!("unknown session: {uuid}")),
              Some(s) => (s.ek.to_bytes(), s.rk.to_bytes()),
          }
      };

      let state_clone = state.clone();
      let result = web::block(move || {
          let params = &state_clone.params;
          let ek = EvaluationKey::from_bytes(&ek_bytes, params)?;
          let rk = RelinearizationKey::from_bytes(&rk_bytes, params)?;
          let responses: anyhow::Result<Vec<Vec<u8>>> = queries.iter()
              .map(|q| process_query(params, &ek, &rk, &state_clone.db,
                                     state_clone.dim1, state_clone.dim2, q))
              .collect();
          responses
      }).await;

      match result {
          Ok(Ok(responses)) => {
              // Pack responses: [num: u32 LE][len: u32 LE][bytes]...
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
      HttpResponse::Ok().json(serde_json::json!({
          "num_tiles": state.num_tiles,
          "tile_size": state.tile_size,
          "dim1": state.dim1,
          "dim2": state.dim2,
          "expansion_level": state.expansion_level,
          "elements_per_plaintext": state.elements_per_pt,
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
      // init tracing, parse CLI, build BFV params, load DB, build ServerState, start HttpServer
      // Pattern identical to spiral-cpu/server/src/main.rs
      // Key differences:
      //   - build BfvParameters via BfvParametersBuilder
      //   - load_db() returns Vec<Plaintext> (not Vec<u64>)
      //   - sessions: RwLock<HashMap<String, SessionKeys>> (no pre-stored params)
      //   - routes: /api/setup, /api/batch-query, /api/params, /api/tile-mapping
  }
  ```

  > **Key note on re-serialising session keys**: Storing `SessionKeys { ek, rk }` directly
  > is preferred, but if `EvaluationKey` is not `Send + Sync`, store the raw bytes
  > (`Vec<u8>`) and deserialise per request (same pattern spiral-cpu uses for
  > `PublicParameters`). Check the fhe crate's trait implementations first.

---

#### Group B: Demo Wiring (Execute B1, B2, B3 in parallel after Group A)

- [x] **Task B1**: Create `mulpir-cpu/demo/frontend/`
  - **Files**: `index.html`, `app.js`

  **`index.html`** — copy `spiral-cpu/demo/frontend/index.html` with:
  - Title: `PIR Map Demo — MulPIR CPU`
  - Subtitle: `Map tiles fetched via BFV MulPIR (CPU)`
  - Badge text: `BFV MulPIR Active`
  - Metrics panel id `cpu-metrics` (identical structure — CPU metrics)
  - Script tag: `<script type="module" src="app.js"></script>`

  **`app.js`** — implement fully:

  ```javascript
  import init, { PIRClient } from './pkg/mulpir_cpu_wasm.js';
  import { LRUTileCache } from '/shared/tile-cache.js';
  import { TileBatchDispatcher } from '/shared/tile-batch.js';
  import { decodeSlotToPBF, decodeMultiSlotToPBF } from '/shared/tile-decoder.js';
  import { initMap } from '/shared/map-setup.js';

  let client = null;
  let sessionUuid = null;
  let tileMapping = null;
  let queryCount = 0;
  let totalLatencyMs = 0;
  let lastQueryMs = 0;
  const tileCache = new LRUTileCache(500 * 1024 * 1024);

  const mulpirCpuBackend = {
      processBatch: async (tiles, abortSignal) => {
          // Build query list: flat list of {tileIdx, pirIdx}
          const queryList = [];
          for (let ti = 0; ti < tiles.length; ti++) {
              for (const pirIdx of tiles[ti].slots) {
                  queryList.push({ tileIdx: ti, pirIdx });
              }
          }

          // Create query bytes for each pirIdx
          const uuidBytes = new TextEncoder().encode(sessionUuid);  // 36 bytes
          const queryParts = queryList.map(q =>
              new Uint8Array(client.create_query(q.pirIdx))
          );

          // Pack batch payload: [UUID:36B][num:4B][q1_len:4B][q1]...[qN_len:4B][qN]
          const totalQueryBytes = queryParts.reduce((s, q) => s + 4 + q.length, 0);
          const payload = new Uint8Array(36 + 4 + totalQueryBytes);
          const view = new DataView(payload.buffer);
          payload.set(uuidBytes, 0);
          view.setUint32(36, queryParts.length, true);
          let off = 40;
          for (const q of queryParts) {
              view.setUint32(off, q.length, true);
              payload.set(q, off + 4);
              off += 4 + q.length;
          }

          // POST /api/batch-query
          const resp = await fetch('/api/batch-query', {
              method: 'POST',
              body: payload,
              headers: { 'Content-Type': 'application/octet-stream' },
              signal: abortSignal,
          });
          if (!resp.ok) throw new Error(`/api/batch-query failed: ${resp.status}`);

          // Parse response: [num:4B][r1_len:4B][r1]...[rN_len:4B][rN]
          const respBuf = new Uint8Array(await resp.arrayBuffer());
          const rv = new DataView(respBuf.buffer);
          const numResp = rv.getUint32(0, true);
          const encryptedResponses = [];
          let roff = 4;
          for (let i = 0; i < numResp; i++) {
              const sz = rv.getUint32(roff, true);
              encryptedResponses.push(respBuf.subarray(roff + 4, roff + 4 + sz));
              roff += 4 + sz;
          }

          // Decrypt, decode slot to PBF, collect results
          const slotParts = tiles.map(() => []);
          for (let i = 0; i < queryList.length; i++) {
              const raw = new Uint8Array(client.decrypt_response(
                  encryptedResponses[i], queryList[i].pirIdx
              ));
              slotParts[queryList[i].tileIdx].push(raw);
          }

          const results = new Map();
          for (let ti = 0; ti < tiles.length; ti++) {
              const tile = tiles[ti];
              let result;
              try {
                  const decoded = slotParts[ti];
                  result = decoded.length > 1
                      ? decodeMultiSlotToPBF(decoded)
                      : decodeSlotToPBF(decoded[0]);
              } catch { result = new ArrayBuffer(0); }
              results.set(tile.key, result);
          }
          return results;
      }
  };

  const dispatcher = new TileBatchDispatcher(mulpirCpuBackend, 50);

  // Speculative prefetch (same as spiral-cpu/demo/frontend/app.js)
  const NEIGHBOR_OFFSETS = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
  function prefetchNeighbors(z, x, y) { /* identical to spiral-cpu */ }

  async function fetchTileViaPIR(z, x, y, abortSignal) { /* identical to spiral-cpu */ }

  function updatePirStats() { /* identical to spiral-cpu */ }

  function startMetricsPolling() { /* identical to spiral-cpu: polls /api/metrics */ }

  async function initialize() {
      // 1. Load WASM: await init()
      // 2. GET /api/params → pirParams (JSON with dim1, dim2, expansion_level, etc.)
      // 3. Create client: new PIRClient(pirParams.num_tiles, pirParams.tile_size)
      // 4. Generate keys with progress:
      //    - setStatus('Generating evaluation key (~5s)...')
      //    - const galoisBytes = new Uint8Array(client.generate_galois_key())
      //    - const relinBytes  = new Uint8Array(client.generate_relin_key())
      // 5. Upload keys: POST /api/setup
      //    - Body: [galois_len:4B LE][galoisBytes][relinBytes]
      //    - Store UUID as sessionUuid
      // 6. GET /api/tile-mapping → tileMapping Map
      // 7. initMap(mappingData, fetchTileViaPIR)
      // 8. Show UI, start metrics polling
  }

  initialize();
  ```

  > **Note on key generation progress**: Galois key generation (5–30 s on CPU) blocks the
  > WASM thread. Call `await new Promise(r => setTimeout(r, 0))` between status updates to
  > keep the loading screen responsive.

---

- [x] **Task B2**: Create `mulpir-cpu/demo/proxy/server.py`
  - **File**: `mulpir-cpu/demo/proxy/server.py`
  - Copy `spiral-cpu/demo/proxy/server.py` wholesale, then change:
    1. Module docstring: replace "Spiral-CPU" → "MulPIR-CPU"
    2. Remove the `/api/private-read` endpoint
    3. Add `/api/setup` (same as spiral-cpu's `/api/setup`):
       - Forwards POST body to `{spiral_base}/api/setup`
       - Returns UUID text/plain
    4. Add `/api/batch-query`:
       - Forwards POST body verbatim to `{spiral_base}/api/batch-query`
       - Returns raw binary response (`content_type="application/octet-stream"`)
       - Timeout: 120 seconds (MulPIR CPU queries can be slow)
    5. Keep `/api/params`, `/api/tile-mapping`, `/api/metrics`, `/shared/`, static routes
       unchanged.
    6. `--spiral-port` default stays 8081.

  **Complete `create_app()` function**:
  ```python
  def create_app(spiral_host, spiral_port, tiles_dir):
      # Same structure as spiral-cpu proxy
      # Routes:
      #   POST /api/setup          → forward to spiral_base/api/setup, return UUID text/plain
      #   POST /api/batch-query    → forward to spiral_base/api/batch-query, return octet-stream
      #   GET  /api/params         → forward to spiral_base/api/params, return JSON
      #   GET  /api/tile-mapping   → load_tile_mapping(tiles_dir), return JSON
      #   GET  /api/metrics        → psutil CPU/memory stats
      #   GET  /shared/<filename>  → pir-map-shared/frontend/<filename>
      #   GET  /                   → index.html
      #   GET  /<path>             → static frontend files
  ```

---

- [x] **Task B3**: Create `mulpir-cpu/run_demo.sh` and update `.gitignore`

  **`mulpir-cpu/run_demo.sh`** — copy `spiral-cpu/run_demo.sh` and change:
  - `MULPIR_PORT=8081` (default server port)
  - `PROXY_PORT=8003` (avoid collision with spiral-cpu on 8002)
  - Step 1: Build `crates/mulpir-cpu-wasm` with wasm-pack:
    ```bash
    WASM_PKG="$ROOT/../crates/mulpir-cpu-wasm/pkg"
    if [[ ! -f "$WASM_PKG/mulpir_cpu_wasm_bg.wasm" ]]; then
        (cd "$ROOT/../crates/mulpir-cpu-wasm" && wasm-pack build --target web --release)
    fi
    ```
  - Step 2: Symlink `$WASM_PKG` → `$ROOT/demo/frontend/pkg`
  - Step 3: Build `mulpir-cpu/server`:
    ```bash
    SERVER="$ROOT/server/target/release/mulpir_cpu_server"
    if [[ ! -x "$SERVER" ]]; then
        (cd "$ROOT/server" && cargo build --release)
    fi
    ```
  - Step 4: Tile detection (identical to spiral-cpu):
    - `--synthetic` flag calls `prepare_tiles.py --synthetic`
    - reads `num_pir_slots` and `tile_size` from tile_mapping.json
  - Step 5: Start `mulpir_cpu_server` with same flags as spiral_server
  - Step 6: Start Flask proxy: `python3 "$ROOT/demo/proxy/server.py" --spiral-port $MULPIR_PORT --port $PROXY_PORT --tiles-dir $TILES_DIR`
  - Step 7: Print URL (`http://localhost:$PROXY_PORT`)
  - Step 8: Optional ngrok

  **`.gitignore`** additions (append to existing file):
  ```gitignore
  # MulPIR CPU server build
  mulpir-cpu/server/target/

  # MulPIR CPU WASM frontend symlinks
  mulpir-cpu/demo/frontend/pkg
  ```

---

## Implementation Workflow

This plan file serves as the authoritative checklist for implementation. When implementing:

### Required Process
1. **Load Plan**: Read this entire plan file before starting.
2. **Sync Tasks**: Create TodoWrite tasks matching the checkboxes above.
3. **Execute & Update**: For each task:
   - Mark TodoWrite as `in_progress` when starting.
   - Update checkbox `[ ]` to `[x]` when completing.
   - Mark TodoWrite as `completed` when done.
4. **Maintain Sync**: Keep this file and TodoWrite synchronised throughout.

### Critical Rules
- This plan file is the source of truth for progress.
- Group A tasks (A1, A2) MUST be completed before Group B tasks (B1, B2, B3).
- Group A tasks run in parallel; Group B tasks run in parallel with each other.
- Tasks should be run in parallel where possible using subtasks to avoid context bloat.
- Mark tasks complete only when fully implemented (no placeholders).
- Before implementing A1, read `crates/fhe-wasm/src/lib.rs` to understand:
  - The exact `EvaluationKeyBuilder` call chain for Galois keys.
  - The `Plaintext` coefficient accessor used in `decrypt_response`.
- Before implementing A2, confirm `EvaluationKey` implements `Send + Sync` (or adapt session
  storage to use raw bytes if it does not).

### Progress Tracking
The checkboxes above represent the authoritative status of each task. Keep them updated as you work.
