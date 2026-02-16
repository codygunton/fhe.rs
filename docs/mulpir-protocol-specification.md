# MulPIR Protocol Specification

## Protocol Overview

MulPIR is a Private Information Retrieval (PIR) protocol built on the BFV (Brakerski-Fan-Vercauteren) fully homomorphic encryption scheme. It enables a client to retrieve a specific element from a server-hosted database without revealing the queried index. The protocol improves upon SealPIR by using multiplicative homomorphic operations to achieve better communication complexity. The security guarantee is computational: an honest-but-curious server learns nothing about which element the client requested.

Reference: https://eprint.iacr.org/2019/1483

---

## Parameter Space

### BFV Encryption Parameters

| Parameter | Symbol | Typical Value | Constraints |
|-----------|--------|---------------|-------------|
| Polynomial degree | `n` | 8192 | Power of 2, >= 8 |
| Plaintext modulus | `t` | 0x1E0001 (1,966,081) | Must support NTT: `t = 1 mod 2n` |
| Ciphertext moduli | `q_i` | [50, 55, 55] bits | Each 10-62 bits |
| Error variance | `sigma^2` | 10 | 1-16 |

**File Reference**: `/home/cody/fhe.rs/crates/fhe/src/bfv/parameters.rs:21-53`

### PIR-Specific Parameters

| Parameter | Description | Derivation |
|-----------|-------------|------------|
| `database_size` | Number of elements | User input |
| `element_size` | Bytes per element | User input, max = `(log2(t) * n) / 8` |
| `elements_per_plaintext` | Packing factor | `(log2(t) * n) / (element_size * 8)` |
| `number_rows` | Packed database rows | `ceil(database_size / elements_per_plaintext)` |
| `dim1`, `dim2` | Matrix dimensions | `dim1 = ceil(sqrt(number_rows))`, `dim2 = ceil(number_rows / dim1)` |
| `expansion_level` | Galois key depth | `ceil(log2(dim1 + dim2))` |

**File Reference**: `/home/cody/fhe.rs/crates/fhe/examples/util.rs:86-134`

---

## Data Structures

### Core Types

#### `BfvParameters`
Central configuration holding all scheme parameters and precomputed values.

```
BfvParameters {
    polynomial_degree: usize,           // n
    plaintext_modulus: u64,             // t
    moduli: Box<[u64]>,                 // [q_0, q_1, ..., q_L]
    context_chain: Arc<ContextLevel>,   // Linked list for modulus switching
    ntt_operator: Option<Arc<NttOperator>>,  // For SIMD encoding
}
```
**File**: `/home/cody/fhe.rs/crates/fhe/src/bfv/parameters.rs:21-53`

#### `Ciphertext`
Encrypted data, represented as a vector of polynomials in NTT form.

```
Ciphertext {
    par: Arc<BfvParameters>,
    c: Vec<Poly>,           // [c_0, c_1] for fresh, [c_0, c_1, c_2] after multiplication
    level: usize,           // Current modulus chain position
    seed: Option<Seed>,     // For compact serialization
}
```
📐 **Shape**: `c[i]` is `(num_moduli, degree)` array of `u64`

**File**: `/home/cody/fhe.rs/crates/fhe/src/bfv/ciphertext.rs:17-30`

#### `Plaintext`
Encoded message ready for encryption or homomorphic operations.

```
Plaintext {
    par: Arc<BfvParameters>,
    value: Box<[u64]>,      // Raw encoded values
    poly_ntt: Poly,         // NTT representation for fast operations
    level: usize,
    encoding: Option<Encoding>,
}
```
📐 **Shape**: `value` is `degree` elements, `poly_ntt` is `(num_moduli, degree)`

**File**: `/home/cody/fhe.rs/crates/fhe/src/bfv/plaintext.rs:13-26`

#### `Poly`
RNS polynomial with multiple representations.

```
Poly {
    ctx: Arc<Context>,
    representation: Representation,     // PowerBasis | Ntt | NttShoup
    coefficients: Array2<u64>,          // (num_moduli, degree)
    coefficients_shoup: Option<Array2<u64>>,  // Precomputed for fast multiply
}
```
📐 **Shape**: `coefficients` is `(L+1, n)` where `L` is the number of moduli minus one

**File**: `/home/cody/fhe.rs/crates/fhe-math/src/rq/mod.rs:86-94`

### Key Types

#### `SecretKey`
```
SecretKey {
    par: Arc<BfvParameters>,
    coeffs: Zeroizing<Box<[i64]>>,  // Ternary coefficients {-1, 0, 1}
}
```

#### `EvaluationKey`
Enables oblivious expansion of ciphertexts.
```
EvaluationKey {
    gk: HashMap<usize, GaloisKey>,  // Exponent -> Galois key mapping
    monomials: Vec<Poly>,           // Precomputed x^(-2^l) for expansion
    ciphertext_level: usize,
    evaluation_key_level: usize,
}
```
**File**: `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/evaluation_key.rs:22-37`

#### `RelinearizationKey`
Reduces ciphertext size after multiplication.
```
RelinearizationKey {
    ksk: KeySwitchingKey,  // Switches from s^2 to s
}
```
**File**: `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/relinearization_key.rs:19-25`

---

## Protocol Phases

### Phase 1: Parameter Generation

**Purpose**: Establish cryptographic parameters ensuring 128-bit security.

**Inputs**: `degree`, `plaintext_modulus`, `moduli_sizes`
**Outputs**: `Arc<BfvParameters>`

**Key Functions**:
- `BfvParametersBuilder::build()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/parameters.rs:378-555`
- `generate_prime()` @ `/home/cody/fhe.rs/crates/fhe-math/src/zq/primes.rs`

**Operations**:
1. Validate degree is power of 2
2. Generate/validate ciphertext moduli (NTT-friendly primes)
3. Build context chain for modulus switching
4. Precompute delta scaling factors: `delta = round(q/t)`
5. Initialize NTT operators for each modulus

---

### Phase 2: Database Preprocessing (Server)

**Purpose**: Pack and encode the database for efficient homomorphic operations.

**Inputs**: `database: Vec<Vec<u8>>`, `params: Arc<BfvParameters>`
**Outputs**: `(Vec<Plaintext>, (dim1, dim2))`

**Key Function**: `encode_database()` @ `/home/cody/fhe.rs/crates/fhe/examples/util.rs:97-134`

**Operations**:
1. Compute packing parameters:
   - `elements_per_plaintext = (log2(t) * n) / (element_size * 8)`
   - `number_rows = ceil(database_size / elements_per_plaintext)`
2. Reshape database into `dim1 x dim2` matrix
3. For each row:
   - Pack multiple elements into byte array
   - Transcode bytes to polynomial coefficients via `transcode_from_bytes()`
   - Encode as `Plaintext` in NTT representation

📐 **Output Shape**: `dim1 * dim2` plaintexts, each encoding up to `elements_per_plaintext` database elements

⚡ **GPU Hotspot**: Encoding loop is embarrassingly parallel across rows

---

### Phase 3: Client Setup

**Purpose**: Generate secret key and evaluation keys for the server.

**Inputs**: `params: Arc<BfvParameters>`, `(dim1, dim2)`
**Outputs**: `(SecretKey, EvaluationKey_bytes, RelinearizationKey_bytes)`

**Key Functions**:
- `SecretKey::random()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/secret_key.rs`
- `EvaluationKeyBuilder::enable_expansion()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/evaluation_key.rs:292-299`
- `RelinearizationKey::new_leveled()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/relinearization_key.rs:34-41`

**Operations**:
1. Sample ternary secret key
2. Compute expansion level: `level = ceil(log2(dim1 + dim2))`
3. Generate Galois keys for indices `{(n >> l) + 1 : l in 0..level}`
4. Generate relinearization key (key-switching from s^2 to s)

⚡ **GPU Hotspot**: Galois key generation involves many polynomial multiplications

---

### Phase 4: Query Generation (Client)

**Purpose**: Create encrypted selection vector for target index.

**Inputs**: `index: usize`, `(dim1, dim2)`, `SecretKey`
**Outputs**: `query_bytes: Vec<u8>`

**Key Function**: Query construction in `main()` @ `/home/cody/fhe.rs/crates/fhe/examples/mulpir.rs:133-148`

**Operations**:
1. Compute query index in packed database: `query_index = index / elements_per_plaintext`
2. Decompose: `i = query_index / dim2`, `j = query_index % dim2`
3. Create selection vector of length `dim1 + dim2`:
   - `pt[i] = (2^level)^{-1} mod t`
   - `pt[dim1 + j] = (2^level)^{-1} mod t`
   - All other positions = 0
4. Encode as polynomial plaintext
5. Encrypt at level 1

📐 **Query Size**: Single ciphertext, ~serialized as `2 * (num_moduli - 1) * degree * 8` bytes (with seed optimization)

---

### Phase 5: Server Response

**Purpose**: Homomorphically compute the requested database element.

**Inputs**: `query_bytes`, `preprocessed_database`, `EvaluationKey`, `RelinearizationKey`
**Outputs**: `response_bytes: Vec<u8>`

**Key Functions**:
- `EvaluationKey::expands()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/evaluation_key.rs:153-193`
- `dot_product_scalar()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/ops/dot_product.rs:56-157`
- `RelinearizationKey::relinearizes()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/relinearization_key.rs:75-104`

#### Step 5.1: Query Expansion

**Purpose**: Expand single ciphertext to `dim1 + dim2` ciphertexts.

```
expanded = ek_expansion.expands(&query, dim1 + dim2)
```

**Algorithm** (Oblivious Expansion from the paper):
```
for l in 0..level:
    monomial = x^{-(n >> l)}      // Precomputed in EvaluationKey
    gk = galois_key[(n >> l) + 1]
    step = 1 << l
    for i in 0..step:
        sub = gk.relinearize(out[i])   // Key switch c(x^{(n>>l)+1}) to c(x)
        out[step + i] = (out[i] - sub) * monomial
        out[i] = out[i] + sub
```

**File**: `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/evaluation_key.rs:166-186`

📐 **Output**: `dim1 + dim2` ciphertexts

⚡ **GPU Hotspot**:
- Galois key switching: `O(level * 2^level)` key switches
- Each key switch involves polynomial multiplication and NTT operations

#### Step 5.2: Inner Product (First Dimension)

**Purpose**: Compute dot product along columns of the database matrix.

```
query_vec = expanded[..dim1]
for i in 0..dim2:
    column = database[i], database[i + dim2], ..., database[i + (dim1-1)*dim2]
    partial[i] = dot_product_scalar(query_vec, column)
```

**Key Function**: `dot_product_scalar()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/ops/dot_product.rs:56-157`

**Implementation Details**:
- Uses fused multiply-accumulate in 128-bit accumulators
- Loop unrolling with 16-element chunks
- Delayed modular reduction for efficiency

```rust
// Core FMA operation (simplified)
for (ciphertext, plaintext) in zip(ct, pt):
    for (acci, ci) in zip(acc, ciphertext):
        for (accij, cij, pij) in zip(acci, ci, plaintext):
            accij += (cij as u128) * (pij as u128)  // Delayed reduction
```

**File**: `/home/cody/fhe.rs/crates/fhe/src/bfv/ops/dot_product.rs:13-50`

📐 **Output**: `dim2` ciphertexts

⚡ **GPU Hotspot**:
- `dim2` independent dot products, each over `dim1` ciphertext-plaintext pairs
- Memory-bound: accessing `dim1 * dim2 * num_moduli * degree * 8` bytes of database
- Compute: `O(dim1 * dim2 * num_moduli * degree)` multiply-accumulates

#### Step 5.3: Selection and Relinearization (Second Dimension)

**Purpose**: Select the target row using the second set of query ciphertexts.

```
out = zero_ciphertext
for (i, ci) in expanded[dim1..].enumerate():
    out += partial[i] * ci
rk.relinearizes(&mut out)
out.switch_to_level(max_level)
```

**Key Operations**:
- Ciphertext-ciphertext multiplication (produces 3-element ciphertext)
- Relinearization (reduces back to 2-element ciphertext)
- Modulus switching (reduces ciphertext modulus for smaller response)

**Multiplication Details** @ `/home/cody/fhe.rs/crates/fhe/src/bfv/ops/mul.rs:159-234`:
1. Extend both ciphertexts to larger modulus basis
2. Compute: `c0' = c0*c0'`, `c1' = c0*c1' + c1*c0'`, `c2' = c1*c1'`
3. Scale down by `t/Q` factor

**Relinearization Details** @ `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/relinearization_key.rs:75-104`:
1. Decompose `c2` into RNS basis
2. Apply key switching: `(c0 + ksk_0 * c2, c1 + ksk_1 * c2)`

📐 **Output**: Single ciphertext at highest level (smallest modulus)

⚡ **GPU Hotspot**:
- `dim2` ciphertext multiplications
- `dim2` additions of 3-element ciphertexts
- 1 relinearization
- 1 modulus switching chain

---

### Phase 6: Answer Extraction (Client)

**Purpose**: Decrypt and extract the requested element.

**Inputs**: `response_bytes`, `SecretKey`, `index`
**Outputs**: `element: Vec<u8>`

**Key Functions**:
- `SecretKey::try_decrypt()` @ `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/secret_key.rs`
- `transcode_to_bytes()` @ `/home/cody/fhe.rs/crates/fhe-util/src/lib.rs`

**Operations**:
1. Deserialize ciphertext
2. Decrypt: `m' = c0 + c1*s` (in NTT domain)
3. Scale and round: `m = round(t/q * m')`
4. Decode polynomial to coefficient vector
5. Transcode coefficients to bytes
6. Extract element at `offset = index % elements_per_plaintext`

---

## Computational Pipeline Summary

```
                          ┌─────────────────────────────────────────┐
                          │           CLIENT SIDE                   │
                          └─────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────┐    ┌──────────────────────────────────────────────┐
│  Secret Key    │───>│  Query: Encrypt selection vector at level 1  │
│  Generation    │    │  Output: 1 ciphertext                        │
└────────────────┘    └──────────────────────────────────────────────┘
        │                                 │
        │                                 ▼
        │             ┌──────────────────────────────────────────────┐
        │             │              NETWORK TRANSFER                │
        │             │  Query: ~200 KB                              │
        │             │  EvalKey: ~50 MB, RelinKey: ~6 MB            │
        │             └──────────────────────────────────────────────┘
        │                                 │
        │                                 ▼
        │             ┌─────────────────────────────────────────────┐
        │             │            SERVER SIDE                      │
        │             └─────────────────────────────────────────────┘
        │                                 │
        │                                 ▼
        │             ┌──────────────────────────────────────────────┐
        │             │  Step 1: EXPANSION                          │
        │             │  Input: 1 ciphertext                        │
        │             │  Output: dim1 + dim2 ciphertexts             │
        │             │  ⚡ O(level * 2^level) key switches         │
        │             └──────────────────────────────────────────────┘
        │                                 │
        │                                 ▼
        │             ┌──────────────────────────────────────────────┐
        │             │  Step 2: DOT PRODUCTS (Dimension 1)         │
        │             │  Input: dim1 ciphertexts, dim1*dim2 ptexts  │
        │             │  Output: dim2 ciphertexts                   │
        │             │  ⚡ O(dim1 * dim2 * n * L) FMAs             │
        │             └──────────────────────────────────────────────┘
        │                                 │
        │                                 ▼
        │             ┌──────────────────────────────────────────────┐
        │             │  Step 3: SELECTION (Dimension 2)            │
        │             │  Input: dim2 ciphertexts from each step     │
        │             │  Output: 1 ciphertext (3 polynomials)       │
        │             │  ⚡ O(dim2) ciphertext multiplications      │
        │             └──────────────────────────────────────────────┘
        │                                 │
        │                                 ▼
        │             ┌──────────────────────────────────────────────┐
        │             │  Step 4: RELINEARIZATION + MOD SWITCH       │
        │             │  Input: 3-poly ciphertext                   │
        │             │  Output: 2-poly ciphertext at last level    │
        │             │  ⚡ 1 key switch + L modulus switches       │
        │             └──────────────────────────────────────────────┘
        │                                 │
        │                                 ▼
        │             ┌──────────────────────────────────────────────┐
        │             │              NETWORK TRANSFER                │
        │             │  Response: ~12 KB                           │
        │             └──────────────────────────────────────────────┘
        │                                 │
        ▼                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  Decrypt + Extract element at offset                               │
└────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Notes

### Memory Access Patterns

**Database Preprocessing**:
- Sequential read of database elements
- Sequential write of encoded plaintexts
- Good cache locality

**Dot Product (`dot_product_scalar`)**:
- Ciphertext access: strided by `dim2` for column extraction
- Plaintext access: sequential within encoded database
- Accumulator: `(2, num_moduli, degree)` tensor in u128
- ⚡ Potential GPU optimization: transpose database for coalesced access

**Key Switching**:
- Random access into key switching key polynomials
- Polynomial-by-polynomial processing
- ⚡ GPU consideration: batch multiple key switches together

### Parallelization Opportunities

| Operation | Parallelism Type | Independence |
|-----------|-----------------|--------------|
| Database encoding | Data parallel | Per-row |
| Query expansion | Limited | Stages are sequential, within-stage parallel |
| Dot products (Step 2) | Data parallel | Per-column |
| Ciphertext multiplications (Step 3) | Pipeline | Can overlap with dot products |
| Relinearization | Internal only | Single operation |

### RNS Representation

All polynomials use Residue Number System representation:
- Each coefficient `a` is stored as `(a mod q_0, a mod q_1, ..., a mod q_L)`
- Enables SIMD-style parallelism across moduli
- 📐 Shape: `(num_moduli, degree)` array per polynomial

### NTT Optimization

**Representations**:
- `PowerBasis`: Standard coefficient form
- `Ntt`: Number-theoretic transform (for fast multiplication)
- `NttShoup`: NTT with precomputed Shoup factors (fastest multiplication)

**Conversion Cost**: O(n log n) per modulus

⚡ GPU consideration: Batch NTT transforms across multiple polynomials

---

## File Index

| Component | File Path |
|-----------|-----------|
| Main example | `/home/cody/fhe.rs/crates/fhe/examples/mulpir.rs` |
| PIR utilities | `/home/cody/fhe.rs/crates/fhe/examples/util.rs` |
| BFV parameters | `/home/cody/fhe.rs/crates/fhe/src/bfv/parameters.rs` |
| Ciphertext | `/home/cody/fhe.rs/crates/fhe/src/bfv/ciphertext.rs` |
| Plaintext | `/home/cody/fhe.rs/crates/fhe/src/bfv/plaintext.rs` |
| Encoding | `/home/cody/fhe.rs/crates/fhe/src/bfv/encoding.rs` |
| Secret key | `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/secret_key.rs` |
| Evaluation key | `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/evaluation_key.rs` |
| Relinearization key | `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/relinearization_key.rs` |
| Galois key | `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/galois_key.rs` |
| Key switching | `/home/cody/fhe.rs/crates/fhe/src/bfv/keys/key_switching_key.rs` |
| Dot product | `/home/cody/fhe.rs/crates/fhe/src/bfv/ops/dot_product.rs` |
| Multiplication | `/home/cody/fhe.rs/crates/fhe/src/bfv/ops/mul.rs` |
| Polynomial ops | `/home/cody/fhe.rs/crates/fhe-math/src/rq/mod.rs` |
| Polynomial arithmetic | `/home/cody/fhe.rs/crates/fhe-math/src/rq/ops.rs` |
