#![allow(missing_docs)]

use std::sync::Arc;

use fhe::bfv::{self, BfvParametersBuilder, Ciphertext, Encoding, Plaintext, SecretKey};
use fhe_math::rq::{self, traits::TryConvertFrom, Representation};
use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter, Serialize};
use fhe_util::{inverse, transcode_to_bytes};
use prost::Message;
use wasm_bindgen::prelude::*;

// ─── BFV parameter constants ────────────────────────────────────────────────

const DEGREE: usize = 8192;
const PLAINTEXT_MODULUS: u64 = (1 << 20) + (1 << 19) + (1 << 17) + (1 << 16) + (1 << 14) + 1;
const Q_MODULI: [u64; 3] = [0x3FFFFFFFFC001, 0x7FFFFFFFFB4001, 0x7FFFFFFFEAC001];
const P_MODULUS: u64 = 0x7FFFFFFFE90001;
const Q_SIZE: usize = 3;
const Q_PRIME_SIZE: usize = 4; // Q_SIZE + 1 P modulus

/// Size of one key-switching key entry for METHOD_I: 2 * Q_SIZE * Q_PRIME_SIZE * DEGREE.
const KSK_ENTRY_SIZE: usize = 2 * Q_SIZE * Q_PRIME_SIZE * DEGREE; // 196608

// ─── HEonGPU serialization constants ────────────────────────────────────────

const HEONGPU_SCHEME_BFV: u8 = 0x01;
const HEONGPU_KS_METHOD_I: u8 = 0x01;
const HEONGPU_STORAGE_HOST: u8 = 0x01;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Compute how many database elements fit in one BFV plaintext.
fn number_elements_per_plaintext(
    degree: usize,
    plaintext_nbits: usize,
    elements_size: usize,
) -> usize {
    (plaintext_nbits * degree) / (elements_size * 8)
}

/// Galois elements for MulPIR query expansion.
/// Matches mulpir-gpu-server/tests/test_helpers.hpp: expansion_galois_elements().
fn expansion_galois_elements(level: usize) -> Vec<usize> {
    (0..level).map(|l| (DEGREE >> l) + 1).collect()
}

/// RNS factors for key switching (METHOD_I).
/// factor[i] = P_MODULUS % Q_MODULI[i] for each Q modulus.
fn compute_rns_factors() -> [u64; Q_SIZE] {
    [
        P_MODULUS % Q_MODULI[0],
        P_MODULUS % Q_MODULI[1],
        P_MODULUS % Q_MODULI[2],
    ]
}

/// Create a "factor polynomial" for decomposition level `decomp_idx`.
///
/// In NTT form: all coefficients at RNS modulus `decomp_idx` are set to
/// `factor_value`, all other moduli are zero. This encodes the RNS lifting
/// scalar as a constant polynomial in the Q_tilda ring.
fn make_factor_poly(
    decomp_idx: usize,
    factor_value: u64,
    ctx: &Arc<rq::Context>,
) -> Result<rq::Poly, String> {
    let mut data = vec![0u64; Q_PRIME_SIZE * DEGREE];
    for k in 0..DEGREE {
        data[decomp_idx * DEGREE + k] = factor_value;
    }
    rq::Poly::try_convert_from(data, ctx, true, Representation::Ntt)
        .map_err(|e| format!("factor poly: {e}"))
}

/// Extract raw i64 secret key coefficients from a fhe.rs SecretKey via protobuf.
fn extract_sk_coefficients(sk: &SecretKey) -> Result<Vec<i64>, String> {
    let sk_bytes = sk.to_bytes();
    let sk_proto = fhe::proto::bfv::SecretKey::decode(&sk_bytes[..])
        .map_err(|e| format!("decode SK proto: {e}"))?;
    Ok(sk_proto.coeffs)
}

/// Create the secret key as a Poly in the given context, in NTT form.
fn sk_as_ntt_poly(coeffs: &[i64], ctx: &Arc<rq::Context>) -> Result<rq::Poly, String> {
    let mut poly = rq::Poly::try_convert_from(coeffs, ctx, false, Representation::PowerBasis)
        .map_err(|e| format!("SK poly: {e}"))?;
    poly.change_representation(Representation::Ntt);
    Ok(poly)
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

    let mut root = 0u64;
    for candidate in 2..m {
        let r = mod_pow(candidate, quotient, m);
        if is_primitive_root(r, degree, m) {
            root = r;
            break;
        }
    }
    assert!(root != 0, "No primitive root found for m={m}, degree={degree}");

    let gen_sq = mod_mul(root, root, m);
    let mut current = root;
    let mut min_root = root;
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
}

impl NttConverter {
    fn new(moduli: &[u64], n: usize) -> Self {
        let n_power = n.ilog2();
        let degree_2n = 2 * n;
        let mut forward_tables = Vec::with_capacity(moduli.len());

        for &m in moduli {
            let psi = find_minimal_primitive_root(degree_2n, m);

            let mut powers = Vec::with_capacity(n);
            let mut p = 1u64;
            for _ in 0..n {
                powers.push(p);
                p = mod_mul(p, psi, m);
            }

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
        let mut coeff_poly = poly.clone();
        coeff_poly.change_representation(Representation::PowerBasis);
        let raw = coeff_poly.coefficients();
        let slice = raw.as_slice().unwrap();

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

// ─── PIRClient ──────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct PIRClient {
    params: Arc<bfv::BfvParameters>,
    sk: SecretKey,
    num_tiles: usize,
    tile_size: usize,
    elements_per_plaintext: usize,
    dim1: usize,
    dim2: usize,
    expansion_level: usize,
    /// Level-0 polynomial context (Q moduli only) for ciphertext operations.
    ctx_level0: Arc<rq::Context>,
    /// Q_tilda context (Q + P moduli) for evaluation key generation.
    ctx_q_tilda: Arc<rq::Context>,
    /// Raw secret key coefficients (i64) for key generation.
    sk_coeffs: Vec<i64>,
    /// NTT converter for fhe.rs → HEonGPU NTT domain conversion.
    ntt_conv: NttConverter,
}

#[wasm_bindgen]
impl PIRClient {
    /// Create a new PIR client with BFV parameters matching the GPU server.
    #[wasm_bindgen(constructor)]
    pub fn new(num_tiles: usize, tile_size: usize) -> Result<PIRClient, JsError> {
        let moduli_sizes = &[50, 55, 55];

        let params = BfvParametersBuilder::new()
            .set_degree(DEGREE)
            .set_plaintext_modulus(PLAINTEXT_MODULUS)
            .set_moduli_sizes(moduli_sizes)
            .build_arc()
            .map_err(|e| JsError::new(&format!("Failed to build BFV params: {e}")))?;

        let plaintext_nbits = PLAINTEXT_MODULUS.ilog2() as usize;
        let elements_per_plaintext =
            number_elements_per_plaintext(DEGREE, plaintext_nbits, tile_size);
        let num_rows = num_tiles.div_ceil(elements_per_plaintext);
        let dim1 = (num_rows as f64).sqrt().ceil() as usize;
        let dim2 = num_rows.div_ceil(dim1);
        let expansion_level = (dim1 + dim2).next_power_of_two().ilog2() as usize;

        let ctx_level0 = params
            .context_at_level(0)
            .map_err(|e| JsError::new(&format!("ctx_at_level(0): {e}")))?
            .clone();

        let q_tilda_moduli = [Q_MODULI[0], Q_MODULI[1], Q_MODULI[2], P_MODULUS];
        let ctx_q_tilda = rq::Context::new_arc(&q_tilda_moduli, DEGREE)
            .map_err(|e| JsError::new(&format!("Q_tilda context: {e}")))?;

        let ntt_conv = NttConverter::new(&q_tilda_moduli, DEGREE);

        let sk = SecretKey::random(&params, &mut rand::rng());
        let sk_coeffs = extract_sk_coefficients(&sk)
            .map_err(|e| JsError::new(&format!("extract SK: {e}")))?;

        Ok(PIRClient {
            params,
            sk,
            num_tiles,
            tile_size,
            elements_per_plaintext,
            dim1,
            dim2,
            expansion_level,
            ctx_level0,
            ctx_q_tilda,
            sk_coeffs,
            ntt_conv,
        })
    }

    /// Return the expansion level needed for key generation.
    pub fn expansion_level(&self) -> usize {
        self.expansion_level
    }

    /// Generate a Galois key in HEonGPU's native binary format.
    ///
    /// The key is generated using RNS decomposition (METHOD_I), matching
    /// HEonGPU's internal key format. The secret key never leaves the client.
    ///
    /// For each Galois element g needed for query expansion:
    ///   KSK[i] = RLWE_{s(X^{g^{-1}})}( P * delta_i * s )
    /// where delta_i adds the factor only at RNS modulus i.
    /// HEonGPU uses the INVERSE galois element for the encryption key.
    pub fn generate_galois_key(&self) -> Result<Vec<u8>, JsError> {
        let galois_elts = expansion_galois_elements(self.expansion_level);
        let factors = compute_rns_factors();
        let mut rng = rand::rng();

        // Secret key as NTT polynomial in Q_tilda ring
        let sk_ntt = sk_as_ntt_poly(&self.sk_coeffs, &self.ctx_q_tilda)
            .map_err(|e| JsError::new(&e))?;

        let mut keys: Vec<(usize, Vec<u64>)> = Vec::with_capacity(galois_elts.len());

        for &g in &galois_elts {
            // HEonGPU uses the INVERSE Galois element: s(X^{g^{-1}}) not s(X^g)
            let g_inv = inverse(g as u64, 2 * DEGREE as u64)
                .ok_or_else(|| JsError::new(&format!("No inverse for g={g} mod {}", 2 * DEGREE)))?
                as usize;
            let sub_exp = rq::SubstitutionExponent::new(&self.ctx_q_tilda, g_inv)
                .map_err(|e| JsError::new(&format!("SubstitutionExponent({g_inv}): {e}")))?;
            let s_g = sk_ntt
                .substitute(&sub_exp)
                .map_err(|e| JsError::new(&format!("substitute({g_inv}): {e}")))?;

            let mut key_data = vec![0u64; KSK_ENTRY_SIZE];

            for i in 0..Q_SIZE {
                let a =
                    rq::Poly::random(&self.ctx_q_tilda, Representation::Ntt, &mut rng);
                let e = rq::Poly::small(&self.ctx_q_tilda, Representation::Ntt, 1, &mut rng)
                    .map_err(|e| JsError::new(&format!("error poly: {e}")))?;

                let factor_poly = make_factor_poly(i, factors[i], &self.ctx_q_tilda)
                    .map_err(|e| JsError::new(&e))?;

                // message = factor[i] * s (original secret key, not permuted)
                let message = &factor_poly * &sk_ntt;

                // c0 = -(s(X^{g^{-1}}) * a + e) + message
                let c0 = &(&(-&(&(&s_g * &a) + &e)) + &message);

                // Convert from fhe.rs NTT → coefficient form → HEonGPU NTT
                let c0_converted = self.ntt_conv.convert_poly_fhe_to_heongpu(c0);
                let c1_converted = self.ntt_conv.convert_poly_fhe_to_heongpu(&a);

                let base = i * 2 * Q_PRIME_SIZE * DEGREE;
                key_data[base..base + Q_PRIME_SIZE * DEGREE]
                    .copy_from_slice(&c0_converted);
                key_data[base + Q_PRIME_SIZE * DEGREE..base + 2 * Q_PRIME_SIZE * DEGREE]
                    .copy_from_slice(&c1_converted);
            }

            keys.push((g, key_data));
        }

        Ok(serialize_galois_key_heongpu(&galois_elts, &keys))
    }

    /// Generate a relinearization key in HEonGPU's native binary format.
    ///
    /// For each RNS modulus i:
    ///   KSK[i] = RLWE_s( P * delta_i * s^2 )
    pub fn generate_relin_key(&self) -> Result<Vec<u8>, JsError> {
        let factors = compute_rns_factors();
        let mut rng = rand::rng();

        let sk_ntt = sk_as_ntt_poly(&self.sk_coeffs, &self.ctx_q_tilda)
            .map_err(|e| JsError::new(&e))?;

        // s^2 in NTT form (element-wise multiplication in NTT = polynomial multiplication)
        let s_sq = &sk_ntt * &sk_ntt;

        let mut key_data = vec![0u64; KSK_ENTRY_SIZE];

        for i in 0..Q_SIZE {
            let a = rq::Poly::random(&self.ctx_q_tilda, Representation::Ntt, &mut rng);
            let e = rq::Poly::small(&self.ctx_q_tilda, Representation::Ntt, 1, &mut rng)
                .map_err(|e| JsError::new(&format!("error poly: {e}")))?;

            let factor_poly = make_factor_poly(i, factors[i], &self.ctx_q_tilda)
                .map_err(|e| JsError::new(&e))?;

            // message = factor[i] * s^2
            let message = &factor_poly * &s_sq;

            // c0 = -(s * a + e) + message
            let c0 = &(&(-&(&(&sk_ntt * &a) + &e)) + &message);

            // Convert from fhe.rs NTT → coefficient form → HEonGPU NTT
            let c0_converted = self.ntt_conv.convert_poly_fhe_to_heongpu(c0);
            let c1_converted = self.ntt_conv.convert_poly_fhe_to_heongpu(&a);

            let base = i * 2 * Q_PRIME_SIZE * DEGREE;
            key_data[base..base + Q_PRIME_SIZE * DEGREE]
                .copy_from_slice(&c0_converted);
            key_data[base + Q_PRIME_SIZE * DEGREE..base + 2 * Q_PRIME_SIZE * DEGREE]
                .copy_from_slice(&c1_converted);
        }

        Ok(serialize_relin_key_heongpu(&key_data))
    }

    /// Encrypt a PIR query for the given tile index.
    ///
    /// Returns the ciphertext serialized in HEonGPU's native binary format.
    pub fn create_query(&self, tile_index: usize) -> Result<Vec<u8>, JsError> {
        let query_index = tile_index / self.elements_per_plaintext;
        let level = self.expansion_level;
        let inv = inverse(1u64 << level, PLAINTEXT_MODULUS)
            .ok_or_else(|| JsError::new("No modular inverse"))?;

        let mut pt = vec![0u64; self.dim1 + self.dim2];
        pt[query_index / self.dim2] = inv;
        pt[self.dim1 + (query_index % self.dim2)] = inv;

        let query_pt = Plaintext::try_encode(&pt, Encoding::poly(), &self.params)
            .map_err(|e| JsError::new(&format!("encode query: {e}")))?;
        let query_ct: Ciphertext = self
            .sk
            .try_encrypt(&query_pt, &mut rand::rng())
            .map_err(|e| JsError::new(&format!("encrypt query: {e}")))?;

        Ok(serialize_ct_heongpu(&query_ct))
    }

    /// Decrypt a PIR response and extract the tile bytes.
    ///
    /// The response is in HEonGPU's native binary format.
    pub fn decrypt_response(
        &self,
        response_bytes: &[u8],
        tile_index: usize,
    ) -> Result<Vec<u8>, JsError> {
        let response = deserialize_ct_heongpu(response_bytes, &self.params, &self.ctx_level0)
            .map_err(|e| JsError::new(&format!("deserialize response: {e}")))?;

        let pt = self
            .sk
            .try_decrypt(&response)
            .map_err(|e| JsError::new(&format!("decrypt: {e}")))?;

        let coeffs = Vec::<u64>::try_decode(&pt, Encoding::poly())
            .map_err(|e| JsError::new(&format!("decode: {e}")))?;
        let bytes = transcode_to_bytes(&coeffs, 20);

        let offset = tile_index % self.elements_per_plaintext;
        let start = offset * self.tile_size;
        let end = start + self.tile_size;

        if end > bytes.len() {
            return Err(JsError::new(&format!(
                "tile offset out of range: need bytes [{start}..{end}] but have {}",
                bytes.len()
            )));
        }
        Ok(bytes[start..end].to_vec())
    }

    /// Return PIR parameters as JSON for the frontend.
    pub fn get_params_json(&self) -> String {
        format!(
            r#"{{"dim1":{},"dim2":{},"expansion_level":{},"num_tiles":{},"elements_per_plaintext":{},"tile_size":{}}}"#,
            self.dim1,
            self.dim2,
            self.expansion_level,
            self.num_tiles,
            self.elements_per_plaintext,
            self.tile_size,
        )
    }
}

// ─── HEonGPU Galois key serialization ───────────────────────────────────────

/// Serialize a Galois key in HEonGPU's native binary format (customized mode).
///
/// Format matches Galoiskey<Scheme::BFV>::save() / load() in HEonGPU.
fn serialize_galois_key_heongpu(
    galois_elts: &[usize],
    keys: &[(usize, Vec<u64>)],
) -> Vec<u8> {
    let galois_elt_zero: i32 = (2 * DEGREE - 1) as i32; // 16383
    let galoiskey_size: u64 = KSK_ENTRY_SIZE as u64;

    // Pre-allocate: header + per-key data + zero key
    let data_per_key = 4 + KSK_ENTRY_SIZE * 8;
    let capacity = 64 + galois_elts.len() * 4 + keys.len() * data_per_key + KSK_ENTRY_SIZE * 8;
    let mut buf = Vec::with_capacity(capacity);

    // Header fields (must match Galoiskey::save order exactly)
    buf.push(HEONGPU_SCHEME_BFV);                                   // scheme_
    buf.push(HEONGPU_KS_METHOD_I);                                  // key_type
    buf.extend_from_slice(&(DEGREE as i32).to_le_bytes());          // ring_size
    buf.extend_from_slice(&(Q_PRIME_SIZE as i32).to_le_bytes());    // Q_prime_size_
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes());          // Q_size_
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes());           // d_
    buf.push(0x01);                                                  // customized = true
    buf.extend_from_slice(&3i32.to_le_bytes());                     // group_order_
    buf.push(HEONGPU_STORAGE_HOST);                                 // storage_type_
    buf.push(0x01);                                                  // galois_key_generated_

    // Custom Galois elements list
    buf.extend_from_slice(&(galois_elts.len() as u32).to_le_bytes());
    for &g in galois_elts {
        buf.extend_from_slice(&(g as u32).to_le_bytes());
    }

    buf.extend_from_slice(&galois_elt_zero.to_le_bytes());          // galois_elt_zero
    buf.extend_from_slice(&galoiskey_size.to_le_bytes());           // galoiskey_size_

    // Key data entries
    buf.extend_from_slice(&(keys.len() as u32).to_le_bytes());     // key_count
    for &(g, ref data) in keys {
        buf.extend_from_slice(&(g as i32).to_le_bytes());           // map key (galois elt)
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    // Zero key data (column rotation key — not used for expansion, fill with zeros)
    buf.extend_from_slice(&vec![0u8; KSK_ENTRY_SIZE * 8]);

    buf
}

// ─── HEonGPU Relin key serialization ────────────────────────────────────────

/// Serialize a relinearization key in HEonGPU's native binary format.
///
/// Format matches Relinkey<Scheme::BFV>::save() / load() in HEonGPU.
fn serialize_relin_key_heongpu(key_data: &[u64]) -> Vec<u8> {
    let relinkey_size: u64 = KSK_ENTRY_SIZE as u64;

    let mut buf = Vec::with_capacity(36 + KSK_ENTRY_SIZE * 8);

    // Header fields (must match Relinkey::save order exactly)
    buf.push(HEONGPU_SCHEME_BFV);                                   // scheme_
    buf.push(HEONGPU_KS_METHOD_I);                                  // key_type
    buf.extend_from_slice(&(DEGREE as i32).to_le_bytes());          // ring_size
    buf.extend_from_slice(&(Q_PRIME_SIZE as i32).to_le_bytes());    // Q_prime_size_
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes());          // Q_size_
    buf.extend_from_slice(&(Q_SIZE as i32).to_le_bytes());           // d_
    buf.extend_from_slice(&(Q_PRIME_SIZE as i32).to_le_bytes());    // d_tilda_
    buf.extend_from_slice(&1i32.to_le_bytes());                     // r_prime_
    buf.push(HEONGPU_STORAGE_HOST);                                 // storage_type_
    buf.push(0x01);                                                  // relin_key_generated_
    buf.extend_from_slice(&relinkey_size.to_le_bytes());            // relinkey_size_

    // Key data
    for &v in key_data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

// ─── HEonGPU Ciphertext binary format helpers ───────────────────────────────

/// Serialize a fhe.rs Ciphertext into HEonGPU's native binary format.
///
/// HEonGPU BFV ciphertexts are stored in coefficient form (not NTT).
/// fhe.rs stores them internally in NTT form, so we convert to PowerBasis
/// before serializing.
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

    for poly in ct.iter() {
        let mut coeff_poly = poly.clone();
        coeff_poly.change_representation(Representation::PowerBasis);
        for &c in coeff_poly.coefficients().as_slice().unwrap() {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }

    buf
}

/// Deserialize a HEonGPU-format ciphertext into a fhe.rs Ciphertext.
///
/// HEonGPU keeps cipher_size=3 after relinearization (stale c2 remains).
/// Its decrypt() only uses c0+c1*s, so we only take the first 2 components.
fn deserialize_ct_heongpu(
    bytes: &[u8],
    params: &Arc<bfv::BfvParameters>,
    ctx: &Arc<rq::Context>,
) -> Result<Ciphertext, String> {
    if bytes.len() < 21 {
        return Err("Ciphertext too short for header".into());
    }

    let scheme = bytes[0];
    if scheme != HEONGPU_SCHEME_BFV {
        return Err(format!("Expected BFV scheme (0x01), got 0x{scheme:02x}"));
    }

    let ring_size = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
    let num_moduli = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
    let cipher_size = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
    let in_ntt = bytes[13] != 0;

    let coeffs_per_poly = num_moduli * ring_size;
    let repr = if in_ntt {
        Representation::Ntt
    } else {
        Representation::PowerBasis
    };

    // Only use first 2 components (HEonGPU may leave stale c2 after relinearization)
    let use_size = cipher_size.min(2);

    let mut polys = Vec::with_capacity(use_size);
    let mut offset = 21;

    for i in 0..cipher_size {
        let mut raw = vec![0u64; coeffs_per_poly];
        for v in raw.iter_mut() {
            *v = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
        }

        if i < use_size {
            let mut poly = rq::Poly::try_convert_from(raw, ctx, false, repr)
                .map_err(|e| format!("Poly construction: {e}"))?;

            if !in_ntt {
                poly.change_representation(Representation::Ntt);
            }

            polys.push(poly);
        }
    }

    Ciphertext::new(polys, params).map_err(|e| format!("Ciphertext construction: {e}"))
}
