# PIR System Testing Assessment

## What We Have

### `test_compat.cu` — Cross-library compatibility suite

The centrepiece is `SharedSecretKeyPIR`: it imports an fhe.rs-generated secret key,
galois key, and relin key into HEonGPU, loads the test-vector tile database, runs the
full GPU PIR pipeline, decrypts the result, and compares output bytes against fhe.rs's
expected values. This is the **gold-standard** test: a parameter mismatch (wrong modulus,
wrong degree, wrong Q primes), an encoding bug, or a key-serialization error will all
cause a byte-level mismatch and fail immediately.

Supporting tests in the same file cover:
- Modular inverse correctness (query encoding scalar)
- Encrypt/decrypt round-trip with the imported key
- Tile polynomial encode/decode round-trip
- Expansion step diagnostic (expanded[0] = Enc(1), rest = Enc(0))
- Ciphertext–plaintext multiply diagnostic

### `test_encoding.cu`

Transcode/encode round-trips at the coefficient level.

### `pir_e2e_test.rs`

Rust-side end-to-end test that runs queries against the live GPU server over TCP,
exercising the full wire protocol and proxy stack.

---

## What the Tests Catch

- **Parameter mismatches** — wrong modulus, degree, or Q primes fail `SharedSecretKeyPIR`
  immediately.
- **Encoding bugs** — wrong polynomial representation → wrong output bytes → test failure.
- **Key serialization bugs** — malformed galois/relin keys → expand/select produce garbage.
- **GPU vs. Rust arithmetic agreement** — the byte-for-byte comparison is a cross-library
  correctness check.

---

## Gaps

### 1. Thin query-index coverage

`SharedSecretKeyPIR` tests a handful of indices (typically 4–10 out of 4213). A bug
that only manifests for certain row/column positions — e.g. an off-by-one in how
`row = query_index / dim2` or `col = query_index % dim2` is computed — could pass
all current tests. Should exercise indices at the boundaries: 0, dim2-1, dim2,
2*dim2-1, last slot, first slot of last row, etc.

### 2. Wire-protocol path not covered by C++ tests

`SharedSecretKeyPIR` bypasses the TCP wire format entirely — it loads test vectors
directly into C++ structs. A serialization bug in `wire_format.cu`
(e.g. wrong endianness in the header, truncated payload) would not be caught.
The Rust `pir_e2e_test.rs` covers this path, but it requires a live GPU server.

### 3. Batch query correctness untested

The batch query handler (`BATCH_QUERY = 0x05`) calls `process_query` N times and
packs the responses. There is no test that verifies N batched responses equal N
individual responses, nor that the batch framing (length prefixes, response count)
is parsed correctly by the client.

### 4. No automated WASM freshness check

If `crates/fhe-wasm/pkg/` is stale (compiled against wrong parameters), the only
detection is visual — the map renders blank. This was hit in practice. An automated
check that hashes or versions the WASM binary against the source parameters would
catch this before it surfaces as a runtime failure.

### 5. Security property not explicitly tested

We verify that correct queries return correct results, but we have no tests that
assert the *absence* of information leakage in our own code (e.g. that the server's
response bytes are independent of which tile was queried, beyond what the FHE scheme
guarantees). This is a hard property to test, but even a smoke-test that queries two
different indices and checks that the ciphertext structure looks identical in size and
format would be a start.

---

## Recommended Additions (Prioritised)

1. **Expand `SharedSecretKeyPIR` query index coverage** — add boundary indices and
   at least one index per quadrant of the dim1×dim2 grid. Low effort, high value.

2. **Wire-protocol unit test** — a C++ gtest that serializes a query, deserializes it,
   runs PIR, serializes the response, and deserializes it back. Removes the dependency
   on a live server for protocol regression testing.

3. **Batch query regression test** — verify that
   `batch_query([i, j]) == [single_query(i), single_query(j)]`.

4. **WASM parameter snapshot test** — a CI step that extracts the embedded parameter
   constants from the `.wasm` binary (or compiles a small test shim) and asserts they
   match the values in `config.hpp`. Prevents stale-WASM bugs from reaching a running
   demo.
