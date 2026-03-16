#![allow(missing_docs)]

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use spiral_rs::arith::*;
use spiral_rs::client::Client;
use spiral_rs::discrete_gaussian::DiscreteGaussian;
use spiral_rs::gadget::build_gadget;
use spiral_rs::number_theory::invert_uint_mod;
use spiral_rs::params::Params;
use spiral_rs::poly::*;
use spiral_rs::util::params_from_json;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// YPIR constants matching ypir/src/scheme.rs
const STATIC_PUBLIC_SEED: [u8; 32] = [0u8; 32];
const SEED_0: u8 = 0;
const STATIC_SEED_2: [u8; 32] = [
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0,
];

fn get_seed(public_seed_idx: u8) -> [u8; 32] {
    let mut seed = STATIC_PUBLIC_SEED;
    seed[0] = public_seed_idx;
    seed
}

/// Condense a PolyMatrixNTT by packing two CRT limbs into one u64.
/// Matches ypir/src/packing.rs condense_matrix.
fn condense_matrix<'a>(params: &'a Params, a: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    let mut res = PolyMatrixNTT::zero(params, a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let a_poly = a.get_poly(i, j);
            for z in 0..params.poly_len {
                res_poly[z] = a_poly[z] | (a_poly[z + params.poly_len] << 32);
            }
        }
    }
    res
}

/// CRT-pack a query vector: pack two CRT residues into one u64.
/// Matches ypir/src/client.rs pack_query.
fn pack_query(params: &Params, query: &[u64]) -> Vec<u64> {
    query
        .iter()
        .map(|x| {
            let crt0 = x % params.moduli[0];
            let crt1 = x % params.moduli[1];
            crt0 | (crt1 << 32)
        })
        .collect()
}

/// Generate a fresh RLWE public key sample (2 x m matrix).
/// Matches ypir/src/client.rs get_fresh_reg_public_key.
fn get_fresh_reg_public_key<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    m: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> PolyMatrixNTT<'a> {
    let mut p = PolyMatrixNTT::zero(params, 2, m);
    let dg = DiscreteGaussian::init(params.noise_width);
    for i in 0..m {
        let a = PolyMatrixRaw::random_rng(params, 1, 1, rng_pub);
        let e = PolyMatrixRaw::noise(params, 1, 1, &dg, rng);
        let b = &sk_reg.ntt() * &a.ntt();
        let b = &e.ntt() + &b;
        let mut sample = PolyMatrixNTT::zero(params, 2, 1);
        sample.copy_into(&(-&a).ntt(), 0, 0);
        sample.copy_into(&b, 1, 0);
        p.copy_into(&sample, 0, i);
    }
    p
}

/// Generate YPIR expansion parameters.
/// Matches ypir/src/client.rs raw_generate_expansion_params.
fn raw_generate_expansion_params<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    num_exp: usize,
    m_exp: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> Vec<PolyMatrixNTT<'a>> {
    let g_exp = build_gadget(params, 1, m_exp);
    let g_exp_ntt = g_exp.ntt();
    let mut res = Vec::new();

    for i in 0..num_exp {
        let t = (params.poly_len / (1 << i)) + 1;
        let tau_sk_reg = automorph_alloc(sk_reg, t);
        let prod = &tau_sk_reg.ntt() * &g_exp_ntt;

        let sample = get_fresh_reg_public_key(params, sk_reg, m_exp, rng, rng_pub);
        let w_exp_i = &sample + &prod.pad_top(1);
        res.push(w_exp_i);
    }

    res
}

/// RLWE-to-LWE: extract the 'b' scalar row from an RLWE ciphertext.
/// Matches ypir/src/client.rs rlwe_to_lwe_last_row.
fn rlwe_to_lwe_last_row<'a>(_params: &'a Params, ct: &PolyMatrixRaw<'a>) -> Vec<u64> {
    ct.get_poly(1, 0).to_vec()
}

/// Concatenate vectors horizontally (interleaved).
/// Matches ypir/src/util.rs concat_horizontal.
fn concat_horizontal(v_a: &[Vec<u64>], a_rows: usize, a_cols: usize) -> Vec<u64> {
    let mut out = vec![0u64; a_rows * a_cols * v_a.len()];
    for i in 0..a_rows {
        for j in 0..a_cols {
            for k in 0..v_a.len() {
                let idx = i * a_cols + j;
                let out_idx = i * a_cols * v_a.len() + k * a_cols + j;
                out[out_idx] = v_a[k][idx];
            }
        }
    }
    out
}

/// Serialize a Vec<PolyMatrixNTT> (condensed row-1s) to bytes for upload.
fn serialize_pub_params(params: &Params, v: &[PolyMatrixNTT]) -> Vec<u8> {
    let mut out = Vec::new();
    for mat in v {
        for i in 0..mat.rows {
            for j in 0..mat.cols {
                let poly = mat.get_poly(i, j);
                // Only write poly_len values (condensed: 2 CRT limbs packed into 1 u64)
                for z in 0..params.poly_len {
                    out.extend_from_slice(&poly[z].to_le_bytes());
                }
            }
        }
    }
    out
}

/// Deserialize modulus-switched RLWE ciphertext bytes back to PolyMatrixRaw.
/// Matches ypir/src/modulus_switch.rs ModulusSwitch::recover.
fn recover_ct<'a>(params: &'a Params, q_1: u64, q_2: u64, ciphertext: &[u8]) -> PolyMatrixRaw<'a> {
    let q_1_bits = (q_2 as f64).log2().ceil() as usize;
    let q_2_bits = (q_1 as f64).log2().ceil() as usize;

    let mut res = PolyMatrixRaw::zero(params, 2, 1);
    let mut bit_offs = 0;
    for z in 0..params.poly_len {
        let val = read_arbitrary_bits(ciphertext, bit_offs, q_1_bits);
        res.data[z] = rescale(val, q_2, params.modulus);
        bit_offs += q_1_bits;
    }
    for z in 0..params.poly_len {
        let val = read_arbitrary_bits(ciphertext, bit_offs, q_2_bits);
        res.data[params.poly_len + z] = rescale(val, q_1, params.modulus);
        bit_offs += q_2_bits;
    }
    res
}

/// Read `num_bits` from a byte array starting at bit offset `bit_offs`.
fn read_arbitrary_bits(data: &[u8], bit_offs: usize, num_bits: usize) -> u64 {
    let mut val = 0u64;
    for i in 0..num_bits {
        let byte_idx = (bit_offs + i) / 8;
        let bit_idx = (bit_offs + i) % 8;
        if byte_idx < data.len() {
            val |= (((data[byte_idx] >> bit_idx) & 1) as u64) << i;
        }
    }
    val
}

#[wasm_bindgen]
pub struct YpirClient {
    client: Client<'static>,
    params: &'static Params,
    /// Number of RLWE plaintext instances per item (= db_cols / poly_len)
    instances: usize,
    /// RLWE reduced moduli for modulus switching
    rlwe_q_prime_1: u64,
    rlwe_q_prime_2: u64,
    /// Size of one modulus-switched RLWE ciphertext in bytes
    ct_bytes: usize,
}

#[wasm_bindgen]
impl YpirClient {
    /// Create a YPIR+SP client from server-provided params JSON and q_prime values.
    #[wasm_bindgen(constructor)]
    pub fn new(params_json: &str, rlwe_q_prime_1: u64, rlwe_q_prime_2: u64) -> YpirClient {
        let params = Box::new(params_from_json(params_json));
        let params: &'static Params = Box::leak(params);

        let mut client = Client::init(params);
        client.generate_secret_keys();

        let instances = params.instances;

        // Compute ciphertext size
        let q_1_bits = (rlwe_q_prime_2 as f64).log2().ceil() as usize;
        let q_2_bits = (rlwe_q_prime_1 as f64).log2().ceil() as usize;
        let total_bits = (q_1_bits + q_2_bits) * params.poly_len;
        let ct_bytes = (total_bits + 7) / 8;

        YpirClient {
            client,
            params,
            instances,
            rlwe_q_prime_1,
            rlwe_q_prime_2,
            ct_bytes,
        }
    }

    /// Generate expansion key parameters and return serialized bytes for /api/setup.
    pub fn generate_keys(&mut self) -> Vec<u8> {
        let params = self.params;
        let sk_reg = self.client.get_sk_reg().clone();

        let pack_pub_params = raw_generate_expansion_params(
            params,
            &sk_reg,
            params.poly_len_log2,
            params.t_exp_left,
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
        );

        // Extract row-1 and condense (halves the data size)
        let mut pack_pub_params_row_1s = Vec::new();
        for pp in &pack_pub_params {
            let row_1 = pp.submatrix(1, 0, 1, pp.cols);
            let condensed = condense_matrix(params, &row_1);
            pack_pub_params_row_1s.push(condensed);
        }

        serialize_pub_params(params, &pack_pub_params_row_1s)
    }

    /// Generate an encrypted query for the given target row index.
    /// Returns CRT-packed query bytes.
    pub fn generate_query(&self, target_row: usize) -> Vec<u8> {
        let params = self.params;
        let db_rows = 1usize << (params.db_dim_1 + params.poly_len_log2);

        let scale_k = params.modulus / params.pt_modulus;
        let inv_poly_len = invert_uint_mod(params.poly_len as u64, params.modulus).unwrap();

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));

        let mut rlwe_cts = Vec::new();
        for i in 0..(1usize << params.db_dim_1) {
            // Build selection polynomial pre-scaled by inv_poly_len
            // (matches ypir/src/client.rs packing=true path)
            let mut scalar = PolyMatrixRaw::zero(params, 1, 1);
            if i == target_row / params.poly_len {
                scalar.data[target_row % params.poly_len] = scale_k;
            }
            let factor_poly = PolyMatrixRaw::single_value(params, inv_poly_len).ntt();
            let scalar_scaled = scalar_multiply_alloc(&factor_poly, &scalar.ntt()).raw();

            // Encrypt the pre-scaled scalar with regular (unscaled) error.
            // We do NOT post-multiply the ciphertext — the `a` component must remain
            // unscaled so the server's offline hint can correctly cancel `a*s`.
            // The native YPIR uses encrypt_matrix_scaled_reg (scales error only),
            // but unscaled error works fine for SimplePIR noise budget.
            let ct = self.client.encrypt_matrix_reg(
                &scalar_scaled.ntt(),
                &mut ChaCha20Rng::from_entropy(),
                &mut rng_pub,
            );

            let ct_raw = ct.raw();
            rlwe_cts.push(ct_raw);
        }

        // Extract 'b' rows and concatenate
        let b_rows: Vec<Vec<u64>> = rlwe_cts
            .iter()
            .map(|ct| rlwe_to_lwe_last_row(params, ct))
            .collect();
        let query_row = concat_horizontal(&b_rows, 1, params.poly_len);
        assert_eq!(query_row.len(), db_rows);

        // CRT-pack the query
        let packed = pack_query(params, &query_row);

        // Serialize to bytes (8 bytes per u64)
        let mut out = Vec::with_capacity(packed.len() * 8);
        for val in &packed {
            out.extend_from_slice(&val.to_le_bytes());
        }
        out
    }

    /// Decode a server response (all instances for one query).
    /// Input: `instances` concatenated modulus-switched RLWE ciphertexts.
    /// Output: raw tile bytes.
    pub fn decode_response(&self, data: &[u8]) -> Vec<u8> {
        let params = self.params;
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;

        let mut all_coeffs = Vec::new();

        for inst in 0..self.instances {
            let ct_start = inst * self.ct_bytes;
            let ct_end = ct_start + self.ct_bytes;
            let ct_data = &data[ct_start..ct_end];

            // Recover the modulus-switched ciphertext
            let ct = recover_ct(params, self.rlwe_q_prime_1, self.rlwe_q_prime_2, ct_data);

            // Decrypt: sk_reg_full * ct (in NTT domain)
            let dec = self.client.decrypt_matrix_reg(&ct.ntt()).raw();

            // Rescale to plaintext modulus
            for z in 0..params.poly_len {
                let val = rescale(dec.data[z], params.modulus, params.pt_modulus);
                all_coeffs.push(val);
            }
        }

        // Pack 14-bit plaintext coefficients into contiguous bytes
        let total_bits = all_coeffs.len() * pt_bits;
        let total_bytes = (total_bits + 7) / 8;
        let mut out = vec![0u8; total_bytes];
        let mut bit_offs = 0;
        for &coeff in &all_coeffs {
            write_bits(&mut out, coeff, bit_offs, pt_bits);
            bit_offs += pt_bits;
        }

        out
    }

    /// Size in bytes of the setup data (expansion key parameters).
    pub fn setup_bytes(&self) -> usize {
        // poly_len_log2 condensed matrices, each 1 x t_exp_left, each element = poly_len u64s
        self.params.poly_len_log2 * self.params.t_exp_left * self.params.poly_len * 8
    }

    /// Size in bytes of a single query.
    pub fn query_bytes(&self) -> usize {
        let db_rows = 1usize << (self.params.db_dim_1 + self.params.poly_len_log2);
        db_rows * 8 // CRT-packed u64s
    }

    /// Size in bytes of a single response (all instances).
    pub fn response_bytes(&self) -> usize {
        self.instances * self.ct_bytes
    }

    /// Number of instances per item.
    pub fn num_instances(&self) -> usize {
        self.instances
    }
}

/// Write `num_bits` of `val` into byte array at `bit_offs`.
fn write_bits(data: &mut [u8], val: u64, bit_offs: usize, num_bits: usize) {
    for i in 0..num_bits {
        let byte_idx = (bit_offs + i) / 8;
        let bit_idx = (bit_offs + i) % 8;
        if byte_idx < data.len() {
            data[byte_idx] |= (((val >> i) & 1) as u8) << bit_idx;
        }
    }
}
