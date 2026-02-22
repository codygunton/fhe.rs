#![allow(missing_docs)]

use spiral_rs::client::Client;
use spiral_rs::util::params_from_json;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// WASM-exposed PIR client backed by the spiral-rs library.
///
/// `params` is leaked as `&'static` so that the `Client<'static>` it
/// contains can be stored in a single struct without lifetime parameters
/// on the WASM-bindgen boundary.
#[wasm_bindgen]
pub struct SpiralClient {
    client: Client<'static>,
    params: &'static spiral_rs::params::Params,
}

#[wasm_bindgen]
impl SpiralClient {
    /// Create a new `SpiralClient` from a JSON params string.
    ///
    /// The params string uses the spiral-rs format, e.g.:
    /// ```json
    /// {"n":2,"nu_1":10,"nu_2":6,"p":512,"q2_bits":21,"t_gsw":10,
    ///  "t_conv":4,"t_exp_left":16,"t_exp_right":56,
    ///  "instances":11,"db_item_size":100000}
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(params_json: &str) -> SpiralClient {
        let params = Box::new(params_from_json(params_json));
        let params: &'static spiral_rs::params::Params = Box::leak(params);
        let client = Client::init(params);
        SpiralClient { client, params }
    }

    /// Generate public parameters (encryption keys). Returns serialized bytes.
    ///
    /// This operation is slow (~500 ms–2 s) and should be called once during
    /// setup. The returned bytes must be sent to the server before any queries.
    pub fn generate_keys(&mut self) -> Vec<u8> {
        let pub_params = self.client.generate_keys();
        pub_params.serialize()
    }

    /// Generate an encrypted query for database index `idx`.
    ///
    /// Returns the serialized query bytes. Prepend a UUID (36 bytes) before
    /// sending to the server if the protocol requires it.
    pub fn generate_query(&self, idx: usize) -> Vec<u8> {
        let query = self.client.generate_query(idx);
        query.serialize()
    }

    /// Decrypt a server response.
    ///
    /// Returns the raw database slot bytes:
    /// `[u32 data_len LE][gzip PBF data][zero padding]`.
    pub fn decode_response(&self, data: &[u8]) -> Vec<u8> {
        self.client.decode_response(data)
    }

    /// Size in bytes of the serialized public parameters.
    pub fn setup_bytes(&self) -> usize {
        self.params.setup_bytes()
    }

    /// Size in bytes of a single serialized query.
    pub fn query_bytes(&self) -> usize {
        self.params.query_bytes()
    }

    /// Total database capacity (number of items the server can store).
    pub fn num_items(&self) -> usize {
        self.params.num_items()
    }
}
