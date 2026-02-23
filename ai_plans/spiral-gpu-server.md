# Spiral GPU Server Implementation Plan

## Executive Summary

The Spiral PIR CPU server (`spiral-cpu/server`) achieves ~100–150 ms per query on a single
core. We want to bring this down with a CUDA implementation targeting the RTX 5090 (Blackwell,
sm_120, 32 GB VRAM, 1,792 GB/s bandwidth) to hit ~10–25 ms per query — a 5–15× improvement.

**The approach:** Build `spiral-gpu/` as a new C++20 + CUDA project following the
`mulpir-gpu-server` pattern. The same `spiral-wasm` WASM client and spiral-cpu frontend are
reused unchanged; only the server changes. All Spiral arithmetic (NTT, gadget decomp,
automorphism, database multiply) is implemented as custom CUDA kernels using the two Spiral
moduli directly, with GPU-NTT (from `~/HEonGPU/thirdparty/GPU-NTT/`) providing the NTT
butterfly primitives.

**Data flow (unchanged from CPU):**

```
Browser                     spiral-gpu/server (CUDA)
──────────                  ───────────────────────────────────────────
POST /api/setup             Deserialize PublicParameters → upload to GPU VRAM
  │ 6.7 MB keys  ──────────►  v_packing, v_expansion_left/right on device
POST /api/private-read      Deserialize Query → upload to GPU
  │ 64 KB query  ──────────►  1. expand_query()      [GPU: automorph + gadget + NTT]
                             2. db_multiply()        [GPU: dot product over 512 entries]
                             3. fold_and_pack()      [GPU: gadget fold + W-matrix multiply]
                             4. encode_response()    [GPU → host: pack 20-bit coefficients]
  │ 64 KB resp.  ◄──────────  response bytes (same format as spiral-rs process_query)
```

**Why not wrap spiral-rs via FFI?** The CPU library has no GPU hooks. Re-implementing in CUDA
gives full control over memory layout, kernel fusion, and streaming.

**Why not HEonGPU BFV?** Spiral uses plain RLWE + gadget operations — not BFV ciphertexts.
HEonGPU's NTT primitives are reusable, but its high-level API doesn't apply.

---

## Goals & Objectives

### Primary Goals
- Process a single Spiral query (nu_1=9, nu_2=4, instances=3) in ≤25 ms wall-clock on RTX 5090
- Produce byte-for-byte identical responses to `spiral-rs::server::process_query()`
- Expose the same REST API as `spiral-cpu/server` so the existing frontend and WASM client work without changes

### Secondary Objectives
- 1.61 GB database fits in GPU VRAM and is uploaded once at startup
- Support concurrent requests (GPU VRAM resident for multiple in-flight queries)
- `run_demo.sh --gpu` parallels the CPU demo script

---

## Solution Overview

### Approach

New project `spiral-gpu/` with a C++20 + CUDA server. Custom CUDA kernels implement every
Spiral operation over the two specific moduli (268369921, 249561089). GPU-NTT butterfly
primitives are used directly. The HTTP layer uses `cpp-httplib` (same as mulpir-gpu-server).
The database is loaded from disk at startup and kept permanently in VRAM.

### Key Components

1. **NTT Kernels** (`kernels/ntt.cu`): 2048-element forward/inverse NTT for both moduli,
   entirely in shared memory (16 KB per modulus). One thread block per polynomial (1024 threads
   × 2 elements). Twiddle factors in constant memory.

2. **Polynomial Operations** (`kernels/poly_ops.cu`): NTT-domain matrix × matrix multiply
   (Barrett reduction with precomputed constants), scalar multiply, add, negate.

3. **Automorphism** (`kernels/automorph.cu`): Galois permutation `X → X^t` — scatter kernel
   with conditional sign flip, one thread per coefficient.

4. **Gadget Decomposition** (`kernels/gadget.cu`): Bit extraction across gadget basis — one
   thread per (row, column, coefficient) triple; embarrassingly parallel.

5. **Database Manager** (`pipeline/database.cu`): Mirrors `load_db_from_seek()` — load tiles,
   encode into NTT-domain u64 CRT-packed words, upload to device once at startup.

6. **Query Expansion** (`pipeline/expand_query.cu`): GPU port of `coefficient_expansion()` in
   `server.rs:19–121`. Tree of 13 levels; within each level all elements are independent CUDA
   stream launches.

7. **Database Multiply** (`pipeline/db_multiply.cu`): GPU port of
   `multiply_reg_by_database()` in `server.rs:155–221`. For each z (2048 threads), reduce over
   512 database entries using shared memory tiling.

8. **Fold + Pack** (`pipeline/fold_pack.cu`): GPU port of `fold_ciphertexts()` +
   `multiply_by_packing_matrix()`. Converts nu_2=4 folding levels + W-matrix packing.

9. **Serialization** (`serialization.cpp`): Parse spiral-rs binary format
   (little-endian u64 arrays); pack response bytes using spiral-rs `encode()` algorithm.

10. **HTTP Server** (`main.cpp`): `cpp-httplib` with same four endpoints as spiral-cpu/server.
    Queries dispatched to `process_query_gpu()` via `std::async`.

### Architecture Diagram

```
spiral-gpu/
├── CMakeLists.txt                      CUDA 12+, C++20, sm_120
├── server/
│   ├── include/
│   │   ├── params.hpp                  SpiralParams struct (port from params.rs)
│   │   ├── types.hpp                   PolyMatrixGPU, CiphertextGPU
│   │   └── serialization.hpp           Binary format constants + parse declarations
│   └── src/
│       ├── main.cpp                    HTTP server (cpp-httplib)
│       ├── serialization.cpp           Parse PublicParameters, Query; encode response
│       ├── kernels/
│       │   ├── ntt.cu + ntt.cuh        2048-pt NTT/INTT, two moduli
│       │   ├── arith.cu + arith.cuh    Barrett reduction, modular ops
│       │   ├── automorph.cu            Galois automorphism
│       │   ├── gadget.cu               Gadget decomposition
│       │   └── poly_ops.cu             Matrix multiply, add, scalar ops (NTT domain)
│       └── pipeline/
│           ├── database.cu             DB load + VRAM upload
│           ├── expand_query.cu         Coefficient expansion tree
│           ├── db_multiply.cu          First-dimension dot product
│           ├── fold_pack.cu            Second-dim fold + packing
│           └── process_query.cu        Top-level pipeline
├── demo/
│   ├── frontend/ → ../../spiral-cpu/demo/frontend   (symlink)
│   ├── proxy/    → ../../spiral-cpu/demo/proxy      (symlink)
│   └── tiles/    (tiles.bin, tile_mapping.json, symlink or copy)
└── run_demo.sh                         Same pattern as spiral-cpu/run_demo.sh
```

### Memory Budget (RTX 5090, 32 GB VRAM)

| Resident | Size | Notes |
|----------|------|-------|
| Database | 1.61 GB | 201,326,592 u64 words, loaded once |
| PublicParameters (per session) | 6.7 MB | v_packing + v_expansion |
| Query working buffers | ~50 MB | Expanded ciphertexts (528 × 2 × 2048 × 2 × 8B) |
| NTT twiddle tables | 32 KB | 2048 u64s × 2 moduli, constant memory |
| **Total** | ~1.67 GB | Well within 32 GB |

### Expected Outcomes

- Single query latency: ≤25 ms (GPU compute) + ~2 ms host overhead = ≤27 ms round-trip
- Response bytes identical to `spiral-rs::process_query()` — verified by E2E test
- Setup upload: ~6.7 MB (same as CPU server, happens once per session)
- Demo accessible at `http://localhost:8082` rendering NYC tiles via Spiral GPU PIR

---

## Implementation Tasks

### CRITICAL IMPLEMENTATION RULES
1. **All responses must be byte-identical to spiral-rs** — verified against the E2E test
   in `sdk/lib/spiral-rs/examples/spiral_e2e_test.rs`
2. **Exact parameter match**: poly_len=2048, moduli=[268369921, 249561089], t_conv=4,
   t_exp_right=56, nu_1=9, nu_2=4, n=2, instances=3, p=256, q2_bits=22
3. **GPU-NTT butterfly primitives** can be taken directly from
   `~/HEonGPU/thirdparty/GPU-NTT/src/include/gpuntt/` — adapt `SmallForwardNTT` /
   `SmallInverseNTT` for 2^11 = 2048 (they currently template on N_power=10; extend to 11)
4. **Database layout must match `load_db_from_seek()`** exactly — z-major CRT-packed u64 format
5. **No HEonGPU BFV types** — use raw device pointers + custom types only

### Visual Dependency Tree

```
spiral-gpu/
├── CMakeLists.txt                      (Task #0)
├── server/include/
│   ├── params.hpp                      (Task #0: SpiralParams, setup_bytes(), response_bytes())
│   ├── types.hpp                       (Task #0: PolyMatrixGPU, device ptr wrappers)
│   └── serialization.hpp               (Task #0: parse declarations)
├── server/src/
│   ├── kernels/
│   │   ├── arith.cuh                   (Task #1: Barrett constants + __device__ ops)
│   │   ├── ntt.cu + ntt.cuh            (Task #1: 2048-pt NTT/INTT, both moduli)
│   │   ├── automorph.cu                (Task #1: scatter permutation kernel)
│   │   ├── gadget.cu                   (Task #1: bit-extraction kernel)
│   │   └── poly_ops.cu                 (Task #1: NTT-domain matrix ops)
│   ├── serialization.cpp               (Task #2: parse PP + Query; encode response)
│   ├── pipeline/
│   │   ├── database.cu                 (Task #3: load_db_from_seek GPU port)
│   │   ├── expand_query.cu             (Task #3: coefficient_expansion GPU port)
│   │   ├── db_multiply.cu              (Task #3: multiply_reg_by_database GPU port)
│   │   ├── fold_pack.cu                (Task #3: fold_ciphertexts + pack GPU port)
│   │   └── process_query.cu            (Task #4: pipeline orchestration)
│   └── main.cpp                        (Task #5: HTTP server)
└── run_demo.sh                         (Task #5)
```

### Execution Plan

---

#### Group A: Foundation (Execute all in parallel)

- [x] **Task #0a**: CMakeLists.txt
  - File: `spiral-gpu/CMakeLists.txt` and `spiral-gpu/server/CMakeLists.txt`
  - Mirrors `mulpir-gpu-server/CMakeLists.txt` structure
  - Implements:
    - `cmake_minimum_required(VERSION 3.26)`, C++20, CUDA 12+
    - `set(CMAKE_CUDA_ARCHITECTURES 120)` for RTX 5090 (fallback auto-detect)
    - `set(CMAKE_CUDA_STANDARD 20)`
    - `HEONGPU_CUDA_ARCH_FORCE_MANUAL=ON` to prevent override
    - Dependency: `add_subdirectory($ENV{HOME}/HEonGPU)` for GPU-NTT headers + RMM
    - Target: `spiral_gpu_server` executable
    - Link: `rmm::rmm_logger_impl` (not `rmm::rmm` — see MEMORY.md)
    - Include: `${HEONGPU_ROOT}/thirdparty/GPU-NTT/src/include` for gpuntt headers
    - `set_target_properties(spiral_gpu_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)`
    - FindThrust.cmake: include `${CUDAToolkit_INCLUDE_DIRS}/cccl` for CUDA 13+ (see MEMORY.md)
    - External: `FetchContent` for `cpp-httplib` and `nlohmann/json`

- [x] **Task #0b**: Core types and parameters header
  - Files: `spiral-gpu/server/include/params.hpp`, `types.hpp`
  - `params.hpp` implements `struct SpiralParams`:
    ```cpp
    struct SpiralParams {
        static constexpr uint32_t POLY_LEN    = 2048;
        static constexpr uint32_t CRT_COUNT   = 2;
        static constexpr uint64_t MODULUS_0   = 268369921ULL;  // 28-bit
        static constexpr uint64_t MODULUS_1   = 249561089ULL;  // 28-bit
        static constexpr uint32_t P           = 256;
        static constexpr uint32_t Q2_BITS     = 22;
        static constexpr uint32_t T_CONV      = 4;
        static constexpr uint32_t T_EXP_LEFT  = 5;
        static constexpr uint32_t T_EXP_RIGHT = 56;
        static constexpr uint32_t T_GSW       = 7;
        static constexpr uint32_t N           = 2;     // matrix dimension

        // Runtime params (set from command line or JSON):
        uint32_t nu_1;       // db_dim_1, default 9
        uint32_t nu_2;       // db_dim_2, default 4
        uint32_t instances;  // default 3
        uint32_t db_item_size; // default 20480

        // Derived:
        uint32_t num_items() const;   // 2^(nu_1 + nu_2)
        size_t   setup_bytes() const; // matches spiral-rs Params::setup_bytes()
        size_t   query_bytes() const; // matches spiral-rs Params::query_bytes()
        size_t   response_bytes() const; // matches spiral-rs response_bytes()
        uint32_t db_dim1() const { return 1u << nu_1; }
        uint32_t db_dim2() const { return 1u << nu_2; }
        uint32_t stop_round() const; // matches spiral-rs stop_round()
    };
    ```
  - `types.hpp` implements:
    - `struct DevicePolyMatrix { uint64_t* d_data; uint32_t rows, cols; }` (owns device memory)
    - `struct CiphertextGPU` (2×1 DevicePolyMatrix, NTT domain)
    - `struct PublicParamsGPU` containing:
      - `std::vector<DevicePolyMatrix> v_packing`   (n×t_conv matrices)
      - `std::vector<std::vector<DevicePolyMatrix>> v_expansion_left`
      - `std::vector<std::vector<DevicePolyMatrix>> v_expansion_right`
    - `DeviceDB` struct holding database device pointer + dimensions
    - Helper: `poly_words(rows, cols) = rows * cols * POLY_LEN * CRT_COUNT`

---

#### Group B: CUDA Kernels (Execute all in parallel, after Group A)

Each kernel file mirrors the corresponding spiral-rs source exactly.
Reference files: `sdk/lib/spiral-rs/src/{ntt.rs, arith.rs, poly.rs, gadget.rs}`.

- [ ] **Task #1a**: Barrett reduction and modular arithmetic (`arith.cuh`)
  - File: `spiral-gpu/server/src/kernels/arith.cuh` (header-only `__device__` functions)
  - Mirrors: `sdk/lib/spiral-rs/src/arith.rs`
  - Implements:
    ```cpp
    // Precomputed Barrett constants (computed once on host, passed to kernels):
    struct BarrettConst {
        uint64_t modulus;
        uint64_t cr0;   // cr_0 from arith.rs
        uint64_t cr1;   // cr_1 from arith.rs
    };
    // Host: compute from modulus (matches arith.rs calc_cr0/calc_cr1)
    BarrettConst make_barrett(uint64_t modulus);

    // Device: fast modular multiplication
    __device__ uint64_t barrett_mul(uint64_t a, uint64_t b, const BarrettConst& bc);
    __device__ uint64_t barrett_reduce(uint64_t a, const BarrettConst& bc);
    __device__ uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod);
    __device__ uint64_t mod_neg(uint64_t a, uint64_t mod);
    ```
  - Precomputed Barrett constants for both moduli stored in `__constant__` memory:
    ```cpp
    __constant__ BarrettConst d_bc0;  // for MODULUS_0
    __constant__ BarrettConst d_bc1;  // for MODULUS_1
    ```
  - CRT pack/unpack helpers (mirror `db_packing` format in server.rs):
    ```cpp
    __device__ uint64_t crt_unpack_lo(uint64_t packed);  // lower 32 bits → mod0 val
    __device__ uint64_t crt_unpack_hi(uint64_t packed);  // upper 32 bits → mod1 val
    ```

- [ ] **Task #1b**: NTT / INTT kernels (`ntt.cu` / `ntt.cuh`)
  - Files: `spiral-gpu/server/src/kernels/ntt.cu`, `ntt.cuh`
  - Mirrors: `sdk/lib/spiral-rs/src/ntt.rs` (functions `ntt_forward`, `ntt_inverse`)
  - Reference: `~/HEonGPU/thirdparty/GPU-NTT/src/include/gpuntt/` — adapt
    `SmallForwardNTT` / `SmallInverseNTT` template from N_power=10 to N_power=11 (2048)
  - Algorithm: Cooley-Tukey butterfly, 11 stages, all in shared memory
  - **Kernel launch config**: one thread block per polynomial, 1024 threads, each thread handles
    2 butterfly positions
  - **Shared memory**: 2048 u64s = 16 KB per block (fits in 128 KB per SM)
  - **Twiddle factors**: stored in `__constant__` memory:
    ```cpp
    // Root-of-unity tables: ψ^k for k=0..2047 for both moduli
    // Computed as: ψ = primitive 4096th root of unity mod q
    // (matches spiral-rs ntt.rs table generation — verify roots)
    __constant__ uint64_t d_ntt_roots_0[2048];  // mod MODULUS_0
    __constant__ uint64_t d_ntt_roots_1[2048];  // mod MODULUS_1
    __constant__ uint64_t d_intt_roots_0[2048]; // inverse NTT roots, mod MODULUS_0
    __constant__ uint64_t d_intt_roots_1[2048]; // inverse NTT roots, mod MODULUS_1
    __constant__ uint64_t d_intt_n_inv_0;       // N^{-1} mod MODULUS_0
    __constant__ uint64_t d_intt_n_inv_1;       // N^{-1} mod MODULUS_1
    ```
  - Root computation: host-side `void compute_ntt_tables(SpiralParams&)` using the
    same primitive root as spiral-rs (see ntt.rs `calc_psi_powers`). Root of unity:
    find primitive root r of Z*_q, then ψ = r^((q-1)/2N).
  - Public API:
    ```cpp
    // Forward NTT: modifies d_poly in-place, POLY_LEN*CRT_COUNT u64s
    // Layout: interleaved CRT — [coeff0_mod0, coeff0_mod1, coeff1_mod0, ...]
    __global__ void ntt_forward(uint64_t* d_poly, uint32_t num_polys);
    __global__ void ntt_inverse(uint64_t* d_poly, uint32_t num_polys);
    // Batched versions (grid over multiple polynomials simultaneously)
    void launch_ntt_batch(uint64_t* d_polys, uint32_t count, cudaStream_t s = 0);
    void launch_intt_batch(uint64_t* d_polys, uint32_t count, cudaStream_t s = 0);
    ```
  - **Memory layout**: interleaved CRT matches spiral-rs `PolyMatrixNTT` layout exactly:
    index for poly[row][col][coeff][crt] = `(row*cols + col)*POLY_LEN*CRT_COUNT + coeff*CRT_COUNT + crt`
  - **Correctness test**: after task completion, run `round_trip_test`: random polynomial →
    NTT → INTT → verify equals original (within mod)

- [ ] **Task #1c**: Automorphism kernel (`automorph.cu`)
  - File: `spiral-gpu/server/src/kernels/automorph.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/poly.rs:automorph_poly` and `automorph`
  - Kernel: one thread per coefficient position i (0..POLY_LEN):
    ```cpp
    __global__ void poly_automorph(
        uint64_t* __restrict__ d_out,   // POLY_LEN * CRT_COUNT
        const uint64_t* __restrict__ d_in,
        uint32_t t,                      // automorphism index
        uint32_t num_polys               // batch over entire matrix
    );
    ```
  - Per-thread computation (mirrors poly.rs exactly):
    ```cpp
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t num = (i * t) / POLY_LEN;
    uint32_t rem = (i * t) % POLY_LEN;
    bool negate = (num % 2 == 1);
    for (uint32_t crt = 0; crt < CRT_COUNT; crt++) {
        uint64_t val = d_in[i * CRT_COUNT + crt];
        uint64_t mod = (crt == 0) ? MODULUS_0 : MODULUS_1;
        d_out[rem * CRT_COUNT + crt] = negate ? mod_neg(val, mod) : val;
    }
    ```
  - Optimization: precompute `(i * t) mod POLY_LEN` and `(i * t) / POLY_LEN` as lookup
    table in shared memory to avoid division (POLY_LEN=2048 fits easily)
  - Batched: process `num_polys` polynomials in parallel (grid = (POLY_LEN/256, num_polys))

- [ ] **Task #1d**: Gadget decomposition kernel (`gadget.cu`)
  - File: `spiral-gpu/server/src/kernels/gadget.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/gadget.rs:gadget_invert_rdim`
  - Kernel: one thread per (output row, column, coefficient) — embarrassingly parallel
    ```cpp
    __global__ void gadget_invert(
        uint64_t* __restrict__ d_out,   // (rdim * num_elems) × cols matrix
        const uint64_t* __restrict__ d_inp,  // rdim × cols matrix
        uint32_t rdim,
        uint32_t cols,
        uint32_t num_elems,  // number of gadget limbs
        uint32_t bits_per,   // bits per limb
        uint64_t mask        // (1 << bits_per) - 1
    );
    ```
  - Each thread extracts one gadget limb for one column/coefficient:
    ```cpp
    // For output element (row j, col i, coeff z, limb k):
    uint64_t val = d_inp[(j * cols + i) * POLY_LEN * CRT_COUNT + z * CRT_COUNT + crt];
    d_out[(j + k * rdim) * cols * POLY_LEN * CRT_COUNT + ...] = (val >> (k * bits_per)) & mask;
    ```
  - `bits_per` computed on host matching `gadget.rs:gadget_base_log` formula
  - `num_elems` = `modulus_log2 / bits_per` matching `build_gadget` in gadget.rs

- [ ] **Task #1e**: Polynomial matrix operations (`poly_ops.cu`)
  - File: `spiral-gpu/server/src/kernels/poly_ops.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/poly.rs:{multiply, add, scalar_multiply}`
  - All operations in NTT domain (pointwise):
    ```cpp
    // NTT-domain matrix multiply: result[m×p] = A[m×n] * B[n×p] (pointwise per coeff)
    __global__ void poly_mat_mul(
        uint64_t* d_out, const uint64_t* d_a, const uint64_t* d_b,
        uint32_t m, uint32_t n, uint32_t p
    );
    // Add: result += a
    __global__ void poly_add_inplace(uint64_t* d_result, const uint64_t* d_a,
                                     uint32_t rows, uint32_t cols);
    // Scalar multiply: result = scalar * poly
    __global__ void poly_scalar_mul(uint64_t* d_out, const uint64_t* d_in,
                                    const uint64_t* d_scalar);
    // Negate: result = -poly (mod)
    __global__ void poly_negate(uint64_t* d_out, const uint64_t* d_in,
                                uint32_t rows, uint32_t cols);
    ```
  - `poly_mat_mul`: grid=(POLY_LEN/32, m, p), each thread computes one coeff of one
    output element as inner product over n rows (reduction in registers since n≤8)
  - Barrett reduction applied after each multiply

---

#### Group C: Serialization (parallel with Group B)

- [ ] **Task #2**: spiral-rs binary format serialization
  - Files: `spiral-gpu/server/include/serialization.hpp`,
    `spiral-gpu/server/src/serialization.cpp`
  - Reference: `sdk/lib/spiral-rs/src/client.rs:PublicParameters::{serialize,deserialize}`
    and `Query::{serialize,deserialize}` and `server.rs:process_query` response encoding
  - **PublicParameters layout** (matches client.rs serialize() exactly):
    ```
    [v_packing bytes]          n*(t_conv) matrices of size n×2, each rows*cols*POLY_LEN*CRT_COUNT u64s
    [v_expansion_left bytes]   (stop_round+1) matrices, each a 1×2 matrix rows=1, cols=2, t_exp_left limbs
    [v_expansion_right bytes]  (stop_round+1) matrices, each 1×2, t_exp_right limbs (may be 0 if version=1)
    ```
    All u64s in native-endian (little-endian on x86).
  - Implements:
    ```cpp
    // Parse setup bytes → GPU PublicParamsGPU
    PublicParamsGPU parse_public_params(const uint8_t* data, size_t len,
                                        const SpiralParams& p);
    // Parse query bytes → GPU buffer
    CiphertextGPU   parse_query(const uint8_t* data, size_t len,
                                const SpiralParams& p);
    // Encode GPU response → bytes matching spiral-rs process_query output
    // Reference: server.rs process_query final encoding loop
    std::vector<uint8_t> encode_response(const DevicePolyMatrix* result_mats,
                                         const SpiralParams& p);
    ```
  - `parse_public_params` uploads each matrix to device memory:
    ```cpp
    // For each matrix in the serialized stream:
    cudaMalloc(&pp.v_packing[i].d_data, poly_words(rows, cols) * sizeof(uint64_t));
    cudaMemcpy(pp.v_packing[i].d_data, src, bytes, cudaMemcpyHostToDevice);
    ```
  - `encode_response` mirrors `server.rs` final quantization:
    ```cpp
    // For each instance/trial result matrix:
    //   1. Inverse NTT on device
    //   2. Copy to host
    //   3. Apply encode() quantization (rescale u64 → u8 via p=256 rounding)
    //      matching poly.rs to_vec(p_bits=8, poly_len) logic
    ```
  - Verify: `parse_public_params(params.serialize())` round-trips correctly

---

#### Group D: Pipeline Components (Execute in parallel, after Groups B + C)

- [ ] **Task #3a**: Database loading (`pipeline/database.cu`)
  - File: `spiral-gpu/server/src/pipeline/database.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/server.rs:load_db_from_seek` exactly
  - Implements:
    ```cpp
    struct DeviceDB {
        uint64_t* d_data;     // device pointer
        size_t    num_words;  // = instances * n^2 * num_items * POLY_LEN (matches CPU)
    };
    DeviceDB load_db_to_gpu(const uint8_t* raw_tiles,   // mmap'd file
                             size_t file_size,
                             const SpiralParams& p);
    ```
  - **Exact layout match with spiral-rs** `calc_index` macro:
    ```cpp
    // calc_index([instance, trial, z, ii, j], [instances, trials, poly_len, num_per, dim0])
    // = instance*trials*poly_len*num_per*dim0 + trial*poly_len*num_per*dim0 + z*num_per*dim0 + ii*dim0 + j
    // This is z-major: coefficient position z is 3rd from outer → enables coalesced GPU access
    ```
  - Implementation:
    1. Allocate host staging buffer matching CPU db layout
    2. For each raw tile: decompose bytes into 20-bit coefficients,
       apply CRT packing (`db_word = (coeff % q0) | ((coeff % q1) << 32)`)
    3. Reorder indices to match `calc_index` ordering (z-outer)
    4. `cudaMemcpy` to device
    5. Optionally apply NTT to each polynomial (if load_db stores in NTT domain — verify vs spiral-rs)
  - **Verification**: load a known 2-tile database, download and compare byte-by-byte with
    CPU spiral-rs `load_db_from_seek()` output

- [ ] **Task #3b**: Coefficient expansion (`pipeline/expand_query.cu`)
  - File: `spiral-gpu/server/src/pipeline/expand_query.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/server.rs:coefficient_expansion` (lines 19–121)
    and `regev_to_gsw` (lines 123–153)
  - Input: `CiphertextGPU ct` (2×1 matrix, NTT domain)
  - Output: `std::vector<CiphertextGPU> v_reg` (2^nu_1 + 2^nu_2 ciphertexts)
  - Algorithm (mirrors CPU exactly):
    ```cpp
    // Phase 1: Left expansion (g = stop_round + 1 = nu_1 + nu_2 levels at most)
    // Level r: 2^r → 2^(r+1) ciphertexts
    for (int r = 0; r < g; r++) {
        int num_in  = 1 << r;
        int num_out = 1 << (r + 1);
        // For each i in [0, num_in): launch automorph + gadget + ntt + multiply kernels
        // Automorphism index: t = POLY_LEN + (1 << r) + 1  (matches server.rs line 44)
        // Key: v_expansion_left[r] from PublicParamsGPU
        // All operations in parallel via cudaStream_t per output element
    }
    ```
  - Each expansion step is:
    1. `poly_negate(v[num_in + i], v[i])` — negate copy
    2. `launch_intt_batch(ct_raw, 1)` — to raw poly
    3. `poly_automorph(ct_auto, ct_raw, t)` — apply automorphism
    4. `launch_ntt_batch(ct_auto, 1)` — back to NTT
    5. `gadget_invert(ginv, ct_auto, t_exp_left)` — decompose
    6. `poly_mat_mul(w_ginv, v_expansion_left[r], ginv)` — key multiply
    7. Combine: `poly_add_inplace(v[i], w_ginv)` and `poly_add_inplace(v[num_in+i], w_ginv_neg)`
  - The regev_to_gsw conversion follows immediately after left expansion using
    `v_expansion_right` keys (mirrors server.rs:regev_to_gsw)

- [ ] **Task #3c**: Database multiply (`pipeline/db_multiply.cu`)
  - File: `spiral-gpu/server/src/pipeline/db_multiply.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/server.rs:multiply_reg_by_database` (lines 155–221)
  - Input: `v_firstdim` (512 ciphertexts, NTT domain), `DeviceDB`
  - Output: `v_intermediate` (2^nu_2 = 16 output ciphertexts per trial, per instance)
  - Kernel:
    ```cpp
    __global__ void db_multiply_kernel(
        uint64_t* __restrict__ d_out,      // [num_per × ct_rows × ct_cols × POLY_LEN × CRT_COUNT]
        const uint64_t* __restrict__ d_query,  // v_firstdim: [dim0 × ct_rows × POLY_LEN × CRT_COUNT]
        const uint64_t* __restrict__ d_db,     // [POLY_LEN × num_per × dim0 × pt_rows] CRT-packed
        uint32_t dim0,                         // 512
        uint32_t num_per                        // 16
    );
    ```
  - Grid: `(POLY_LEN, num_per)` — each thread block handles one z, one output slot
  - Each block: load 512 query values + 512 db values into shared memory tiles,
    accumulate 4 sums (2 rows × 2 moduli), write to output
  - Exactly mirrors the sums_out_n0_0/n0_1/n1_0/n1_1 accumulation in CPU code
  - CRT-packed db words unpacked per-thread via `crt_unpack_lo` / `crt_unpack_hi`
  - Barrett reduction applied to each accumulator before write

- [ ] **Task #3d**: Fold + pack (`pipeline/fold_pack.cu`)
  - File: `spiral-gpu/server/src/pipeline/fold_pack.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/server.rs:fold_ciphertexts` and
    `server.rs:multiply_by_packing_matrix`
  - Folding: nu_2=4 iterations, each halves the ciphertext count:
    ```cpp
    void fold_ciphertexts_gpu(
        std::vector<CiphertextGPU>& v,          // modified in-place: 16 → 8 → 4 → 2 → 1
        const std::vector<CiphertextGPU>& v_gsw, // GSW ciphertexts from expand_query
        const SpiralParams& p
    );
    ```
    - Each fold step: `v[i] = v[2i] + GSW_ext_product(v_gsw[k], v[2i+1])`
    - GSW external product = gadget decompose ct + poly mat mul with key
  - Packing: apply v_packing matrices to convert n^2 results to single ciphertext:
    ```cpp
    void pack_gpu(CiphertextGPU& out, const std::vector<CiphertextGPU>& results,
                  const PublicParamsGPU& pp, const SpiralParams& p);
    ```
    Mirrors server.rs `pack` function exactly

---

#### Group E: Top-Level Pipeline (after Group D)

- [ ] **Task #4**: process_query top-level (`pipeline/process_query.cu`)
  - File: `spiral-gpu/server/src/pipeline/process_query.cu`
  - Mirrors: `sdk/lib/spiral-rs/src/server.rs:process_query` (lines 650–741)
  - Implements:
    ```cpp
    std::vector<uint8_t> process_query_gpu(
        const SpiralParams& p,
        const PublicParamsGPU& pp,
        const uint8_t* query_bytes,
        const DeviceDB& db,
        cudaStream_t stream = 0
    );
    ```
  - Orchestration:
    1. `parse_query(query_bytes, ...)` → `CiphertextGPU ct`
    2. `expand_query(ct, pp)` → `v_reg[0..dim1)`, `v_gsw[0..dim2)`
    3. For each `instance` (0..3) and `trial` (0..n^2=4):
       - `db_multiply(v_reg, db, instance, trial)` → `v_folded`
       - `fold_ciphertexts(v_folded, v_gsw)` → single ciphertext
       - `pack(packed, pp)` → result matrix
    4. `encode_response(results)` → `std::vector<uint8_t>`
    5. `cudaStreamSynchronize(stream)`

  - **E2E correctness test**: link against `spiral_e2e_test.rs` test vectors — generate
    public params + query + expected response bytes from Rust, feed into GPU pipeline,
    assert exact byte match
  - Add benchmark: time each phase separately and print to stderr

---

#### Group F: HTTP Server + Demo (after Group E)

- [ ] **Task #5a**: HTTP server (`main.cpp`)
  - File: `spiral-gpu/server/src/main.cpp`
  - Mirrors `spiral-cpu/server/src/main.rs` structure, using `cpp-httplib` instead of actix-web
  - CLI args: `--database`, `--tile-mapping`, `--num-tiles`, `--tile-size`, `--port`
    (same as spiral-cpu/server)
  - Startup:
    1. Parse CLI, compute SpiralParams via `select_params(num_tiles, tile_size)`
    2. mmap database file, `load_db_to_gpu()`
    3. Read tile_mapping.json
    4. Start `httplib::Server` on port
  - Endpoints (byte-identical behavior to spiral-cpu/server):
    - `POST /api/setup` → `parse_public_params()`, store in `sessions` map by UUID,
      return UUID string
    - `POST /api/private-read` → UUID prefix lookup, `process_query_gpu()`,
      return raw response bytes
    - `GET /api/params` → JSON with `num_tiles`, `tile_size`, `spiral_params` JSON,
      `setup_bytes`, `query_bytes`, `num_items`
    - `GET /api/tile-mapping` → raw tile_mapping JSON
    - `GET /api/metrics` → CPU/GPU utilization JSON (add `gpu_util_percent`)
  - Session map: `std::unordered_map<std::string, PublicParamsGPU>` protected by
    `std::shared_mutex` (same semantics as RwLock in Rust)
  - Thread pool: `httplib::Server::new_task_queue = 16` — serialize GPU calls with a
    `std::mutex` (one query at a time initially; can parallelize later via streams)

  - `select_params()` function — mirrors spiral-cpu/server `select_params_json()` exactly:
    ```cpp
    std::string select_params_json(size_t num_tiles, size_t tile_size) {
        // Same nu_2 selection: 2 if num_tiles ≤ 2^11, 4 if ≤ 2^13, else 6
        // Returns JSON string with t_conv=4, t_exp_right=56 (NO version=1)
    }
    ```

- [ ] **Task #5b**: Demo runner (`run_demo.sh`)
  - File: `spiral-gpu/run_demo.sh`
  - Mirrors `spiral-cpu/run_demo.sh` structure with:
    - Builds `spiral_gpu_server` with cmake instead of cargo
    - Same tile loading logic (shared with spiral-cpu)
    - Frontend: symlink to `../spiral-cpu/demo/frontend` (same WASM + JS)
    - Proxy: symlink to `../spiral-cpu/demo/proxy/server.py`
    - Default port: `--port 8082` (avoid collision with 8081 CPU server)
    - `--synthetic` and `--ngrok` flags same as CPU demo

---

## Implementation Workflow

This plan file serves as the authoritative checklist for implementation.

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
- Tasks in the same Group run in parallel via subtasks to avoid context bloat
- **Correctness gate**: after Task #4, run the E2E test before proceeding to Task #5

### Correctness Verification Strategy
1. After Task #1b (NTT): round-trip test (random poly → NTT → INTT = original)
2. After Task #2 (serialization): parse known PP bytes, re-serialize, compare
3. After Task #4 (full pipeline): generate test vectors from
   `sdk/lib/spiral-rs/examples/spiral_e2e_test.rs` and verify GPU output matches exactly
4. After Task #5 (server): run the spiral-cpu e2e test pointing at the GPU server

### Progress Tracking
The checkboxes above represent the authoritative status of each task.
