#![allow(missing_docs)]

use std::sync::Arc;

use fhe::bfv::{
    BfvParametersBuilder, Ciphertext, Encoding, EvaluationKey, EvaluationKeyBuilder, Plaintext,
    RelinearizationKey, SecretKey,
};
use fhe_traits::{DeserializeParametrized, FheDecoder, FheDecrypter, FheEncoder, FheEncrypter, Serialize};
use fhe_util::{inverse, transcode_to_bytes};
use wasm_bindgen::prelude::*;

const DEGREE: usize = 8192;
const PLAINTEXT_MODULUS: u64 = (1 << 20) + (1 << 19) + (1 << 17) + (1 << 16) + (1 << 14) + 1;
const MODULI_SIZES: [usize; 3] = [50, 55, 55];
const BITS_PER_COEFF: usize = 20;

fn build_params() -> Result<Arc<fhe::bfv::BfvParameters>, JsError> {
    BfvParametersBuilder::new()
        .set_degree(DEGREE)
        .set_plaintext_modulus(PLAINTEXT_MODULUS)
        .set_moduli_sizes(&MODULI_SIZES)
        .build_arc()
        .map_err(|e| JsError::new(&format!("BFV params: {e}")))
}

fn compute_dims(num_tiles: usize, tile_size: usize) -> (usize, usize, usize, usize) {
    let elements_per_pt =
        ((PLAINTEXT_MODULUS.ilog2() as usize * DEGREE) / (tile_size * 8)).max(1);
    let num_rows = num_tiles.div_ceil(elements_per_pt);
    let dim1 = (num_rows as f64).sqrt().ceil() as usize;
    let dim2 = num_rows.div_ceil(dim1);
    let expansion_level = (dim1 + dim2).next_power_of_two().ilog2() as usize;
    (dim1, dim2, expansion_level, elements_per_pt)
}

#[wasm_bindgen]
pub struct PIRClient {
    params: Arc<fhe::bfv::BfvParameters>,
    sk: SecretKey,
    ek: EvaluationKey,
    rk: RelinearizationKey,
    dim1: usize,
    dim2: usize,
    expansion_level: usize,
    elements_per_pt: usize,
    tile_size: usize,
}

#[wasm_bindgen]
impl PIRClient {
    #[wasm_bindgen(constructor)]
    pub fn new(num_tiles: usize, tile_size: usize) -> Result<PIRClient, JsError> {
        console_error_panic_hook::set_once();
        let params = build_params()?;
        let mut rng = rand::rng();
        let (dim1, dim2, expansion_level, elements_per_pt) = compute_dims(num_tiles, tile_size);

        let sk = SecretKey::random(&params, &mut rng);

        let ek = EvaluationKeyBuilder::new_leveled(&sk, 1, 0)
            .map_err(|e| JsError::new(&format!("EK builder: {e}")))?
            .enable_expansion(expansion_level)
            .map_err(|e| JsError::new(&format!("enable expansion: {e}")))?
            .build(&mut rng)
            .map_err(|e| JsError::new(&format!("build EK: {e}")))?;

        let rk = RelinearizationKey::new_leveled(&sk, 1, 1, &mut rng)
            .map_err(|e| JsError::new(&format!("RK: {e}")))?;

        Ok(PIRClient {
            params,
            sk,
            ek,
            rk,
            dim1,
            dim2,
            expansion_level,
            elements_per_pt,
            tile_size,
        })
    }

    pub fn expansion_level(&self) -> usize {
        self.expansion_level
    }

    pub fn generate_galois_key(&self) -> Vec<u8> {
        self.ek.to_bytes()
    }

    pub fn generate_relin_key(&self) -> Vec<u8> {
        self.rk.to_bytes()
    }

    pub fn create_query(&self, tile_index: usize) -> Result<Vec<u8>, JsError> {
        let mut rng = rand::rng();
        let level = self.expansion_level as u32;
        let inv = inverse(1u64 << level, PLAINTEXT_MODULUS)
            .ok_or_else(|| JsError::new("No modular inverse for query"))?;
        let slot = tile_index / self.elements_per_pt;
        let row = slot / self.dim2;
        let col = slot % self.dim2;
        let mut pt_vec = vec![0u64; self.dim1 + self.dim2];
        pt_vec[row] = inv;
        pt_vec[self.dim1 + col] = inv;
        let pt = Plaintext::try_encode(&pt_vec, Encoding::poly_at_level(1), &self.params)
            .map_err(|e| JsError::new(&format!("encode query: {e}")))?;
        let ct: Ciphertext = self
            .sk
            .try_encrypt(&pt, &mut rng)
            .map_err(|e| JsError::new(&format!("encrypt query: {e}")))?;
        Ok(ct.to_bytes())
    }

    pub fn decrypt_response(
        &self,
        response_bytes: &[u8],
        tile_index: usize,
    ) -> Result<Vec<u8>, JsError> {
        let ct = Ciphertext::from_bytes(response_bytes, &self.params)
            .map_err(|e| JsError::new(&format!("deserialize response: {e}")))?;
        let pt = self
            .sk
            .try_decrypt(&ct)
            .map_err(|e| JsError::new(&format!("decrypt: {e}")))?;
        let coeffs = Vec::<u64>::try_decode(&pt, Encoding::poly_at_level(2))
            .map_err(|e| JsError::new(&format!("decode: {e}")))?;
        let all_bytes = transcode_to_bytes(&coeffs, BITS_PER_COEFF);
        let offset = tile_index % self.elements_per_pt;
        let start = offset * self.tile_size;
        let end = start + self.tile_size;
        if end > all_bytes.len() {
            return Err(JsError::new(&format!(
                "tile offset out of range: need [{start}..{end}] but have {}",
                all_bytes.len()
            )));
        }
        Ok(all_bytes[start..end].to_vec())
    }

    pub fn get_params_json(&self) -> String {
        format!(
            r#"{{"dim1":{},"dim2":{},"expansion_level":{},"num_tiles":0,"elements_per_plaintext":{},"tile_size":{}}}"#,
            self.dim1,
            self.dim2,
            self.expansion_level,
            self.elements_per_pt,
            self.tile_size,
        )
    }
}
