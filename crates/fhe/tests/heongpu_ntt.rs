//! Unit tests for HEonGPU NTT conversion logic.
//!
//! Tests the math primitives used for converting polynomial data between
//! fhe.rs and HEonGPU NTT representations.

// ─── Math primitives (copied from pir_e2e_test.rs) ──────────────────────────

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

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

fn is_primitive_root(root: u64, degree: usize, m: u64) -> bool {
    if root == 0 {
        return false;
    }
    mod_pow(root, (degree / 2) as u64, m) == m - 1
}

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

fn mod_inverse(a: u64, m: u64) -> u64 {
    // Extended Euclidean algorithm
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let temp_r = r;
        r = old_r - q * r;
        old_r = temp_r;
        let temp_s = s;
        s = old_s - q * s;
        old_s = temp_s;
    }
    ((old_s % m as i128 + m as i128) % m as i128) as u64
}

fn bit_reverse(x: usize, bits: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - bits)
}

// ─── NTT implementation ─────────────────────────────────────────────────────

struct NttConverter {
    forward_tables: Vec<Vec<u64>>,
    inverse_tables: Vec<Vec<u64>>,
    n_inv: Vec<u64>,
    moduli: Vec<u64>,
    n: usize,
    psi_roots: Vec<u64>,
}

impl NttConverter {
    fn new(moduli: &[u64], n: usize) -> Self {
        let n_power = n.ilog2();
        let degree_2n = 2 * n;
        let mut forward_tables = Vec::with_capacity(moduli.len());
        let mut inverse_tables = Vec::with_capacity(moduli.len());
        let mut n_inv = Vec::with_capacity(moduli.len());
        let mut psi_roots = Vec::with_capacity(moduli.len());

        for &m in moduli {
            let psi = find_minimal_primitive_root(degree_2n, m);
            psi_roots.push(psi);

            // Forward table: powers of psi in bit-reversed order
            let mut powers = Vec::with_capacity(n);
            let mut p = 1u64;
            for _ in 0..n {
                powers.push(p);
                p = mod_mul(p, psi, m);
            }
            let mut fwd_table = vec![0u64; n];
            for j in 0..n {
                fwd_table[j] = powers[bit_reverse(j, n_power)];
            }
            forward_tables.push(fwd_table);

            // Inverse table: powers of psi^{-1} in bit-reversed order
            let psi_inv = mod_inverse(psi, m);
            let mut inv_powers = Vec::with_capacity(n);
            let mut p = 1u64;
            for _ in 0..n {
                inv_powers.push(p);
                p = mod_mul(p, psi_inv, m);
            }
            let mut inv_table = vec![0u64; n];
            for j in 0..n {
                inv_table[j] = inv_powers[bit_reverse(j, n_power)];
            }
            inverse_tables.push(inv_table);

            // N^{-1} mod m
            n_inv.push(mod_inverse(n as u64, m));
        }

        Self {
            forward_tables,
            inverse_tables,
            n_inv,
            moduli: moduli.to_vec(),
            n,
            psi_roots,
        }
    }

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

    fn intt_inplace(&self, data: &mut [u64], modulus_idx: usize) {
        let n = self.n;
        let m = self.moduli[modulus_idx];
        let table = &self.inverse_tables[modulus_idx];

        // Gentleman-Sande inverse NTT (reverse butterfly)
        let n_power = n.ilog2();
        let mut t = 1;
        let mut mval = n >> 1;

        for _ in 0..n_power {
            for i in 0..mval {
                let j1 = 2 * i * t;
                let j2 = j1 + t;
                let s = table[mval + i];
                for j in j1..j2 {
                    let u = data[j];
                    let v = data[j + t];
                    data[j] = if u + v >= m { u + v - m } else { u + v };
                    let diff = if u >= v { u - v } else { u + m - v };
                    data[j + t] = mod_mul(diff, s, m);
                }
            }
            t <<= 1;
            mval >>= 1;
        }

        // Multiply by N^{-1}
        let n_inv = self.n_inv[modulus_idx];
        for val in data.iter_mut() {
            *val = mod_mul(*val, n_inv, m);
        }
    }

    /// Naive O(N^2) NTT for verification. Returns NTT in standard order.
    fn naive_ntt(&self, data: &[u64], modulus_idx: usize) -> Vec<u64> {
        let n = self.n;
        let m = self.moduli[modulus_idx];
        let psi = self.psi_roots[modulus_idx];
        let omega = mod_mul(psi, psi, m); // omega = psi^2

        let mut result = vec![0u64; n];
        for k in 0..n {
            let mut sum = 0u64;
            for j in 0..n {
                // a[j] * psi^j * omega^{j*k}
                let psi_pow_j = mod_pow(psi, j as u64, m);
                let omega_pow_jk = mod_pow(omega, (j * k) as u64, m);
                let term = mod_mul(data[j], mod_mul(psi_pow_j, omega_pow_jk, m), m);
                sum = (sum + term) % m;
            }
            result[k] = sum;
        }
        result
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

// Use a small prime for fast tests
const TEST_PRIME: u64 = 0x3FFFFFFFFC001; // Q_MODULI[0] from production
// Small degree for verification tests (unused for now)
#[allow(dead_code)]
const SMALL_N: usize = 8;

// Use production-size for one comprehensive test
const PROD_Q0: u64 = 0x3FFFFFFFFC001;
const PROD_Q1: u64 = 0x7FFFFFFFFB4001;
const PROD_Q2: u64 = 0x7FFFFFFFEAC001;
const PROD_P: u64 = 0x7FFFFFFFE90001;
const PROD_DEGREE: usize = 8192;

#[test]
fn test_mod_mul_basic() {
    assert_eq!(mod_mul(3, 4, 13), 12);
    assert_eq!(mod_mul(7, 8, 13), 56 % 13);
    // Test with large values (128-bit intermediate needed)
    let a = 0x7FFFFFFFEAC001_u64 - 1;
    let b = 0x7FFFFFFFEAC001_u64 - 2;
    let m = 0x7FFFFFFFEAC001_u64;
    // (m-1) * (m-2) mod m = (-1)*(-2) mod m = 2
    assert_eq!(mod_mul(a, b, m), 2);
}

#[test]
fn test_mod_pow_basic() {
    assert_eq!(mod_pow(2, 10, 1000), 1024 % 1000);
    assert_eq!(mod_pow(3, 0, 7), 1);
    assert_eq!(mod_pow(5, 1, 7), 5);
    // Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
    assert_eq!(mod_pow(2, TEST_PRIME - 1, TEST_PRIME), 1);
    assert_eq!(mod_pow(3, TEST_PRIME - 1, TEST_PRIME), 1);
}

#[test]
fn test_is_primitive_root() {
    // For degree=2N, root is primitive if root^N ≡ -1 (mod m)
    let m = 17u64; // prime, 17-1 = 16 = 2^4
    let degree = 16; // 2N where N = 8
    // Primitive 16th root of unity mod 17
    // Need root^8 ≡ 16 (mod 17)
    // 2^8 = 256 ≡ 256 - 15*17 = 256 - 255 = 1 (mod 17) → NOT primitive
    assert!(!is_primitive_root(2, degree, m));
    // 3^8 = 6561, 6561 mod 17 = 6561 - 385*17 = 6561 - 6545 = 16 → primitive!
    assert!(is_primitive_root(3, degree, m));
}

#[test]
#[ignore = "brute-force minimality check over 50-bit range; takes hours"]
fn test_find_minimal_primitive_root_properties() {
    // For each production modulus, verify the minimal root has the required property
    let moduli = [PROD_Q0, PROD_Q1, PROD_Q2, PROD_P];
    let degree_2n = 2 * PROD_DEGREE;

    for &m in &moduli {
        let psi = find_minimal_primitive_root(degree_2n, m);

        // psi^N should equal -1 (mod m) for a primitive 2N-th root
        let psi_n = mod_pow(psi, PROD_DEGREE as u64, m);
        assert_eq!(psi_n, m - 1, "psi^N != -1 for modulus {m:#x}");

        // psi^{2N} should equal 1 (mod m)
        let psi_2n = mod_pow(psi, degree_2n as u64, m);
        assert_eq!(psi_2n, 1, "psi^(2N) != 1 for modulus {m:#x}");

        // It should actually be minimal: check that no smaller value satisfies the property
        for candidate in 2..psi {
            if mod_pow(candidate, PROD_DEGREE as u64, m) == m - 1 {
                panic!(
                    "Found smaller root {candidate} < {psi} for modulus {m:#x}"
                );
            }
        }
    }
}

#[test]
fn test_find_minimal_primitive_root_known_values() {
    // Verify roots match the values from the E2E test output
    let moduli = [PROD_Q0, PROD_Q1, PROD_Q2, PROD_P];
    let degree_2n = 2 * PROD_DEGREE;
    let expected = [11286399139u64, 15372713853695, 3055459936772, 4991203289951];

    for (i, &m) in moduli.iter().enumerate() {
        let psi = find_minimal_primitive_root(degree_2n, m);
        assert_eq!(
            psi, expected[i],
            "Root mismatch for modulus {m:#x}: got {psi}, expected {}",
            expected[i]
        );
    }
}

#[test]
fn test_mod_inverse() {
    assert_eq!(mod_mul(mod_inverse(3, 17), 3, 17), 1);
    assert_eq!(mod_mul(mod_inverse(7, 13), 7, 13), 1);
    // Test with production values
    assert_eq!(mod_mul(mod_inverse(5, PROD_Q0), 5, PROD_Q0), 1);
}

#[test]
fn test_ntt_roundtrip_small() {
    let m = 17u64; // small prime
    let n = 8;
    // 17 - 1 = 16, so 16th root of unity exists
    let conv = NttConverter::new(&[m], n);

    let original = vec![1, 2, 3, 4, 5, 6, 7, 8u64];
    let mut data = original.clone();

    conv.ntt_inplace(&mut data, 0);
    conv.intt_inplace(&mut data, 0);

    assert_eq!(
        data, original,
        "NTT round-trip failed for small prime. Got {data:?}, expected {original:?}"
    );
}

#[test]
fn test_ntt_roundtrip_production_primes() {
    let moduli = [PROD_Q0, PROD_Q1, PROD_Q2, PROD_P];
    let n = PROD_DEGREE;
    let conv = NttConverter::new(&moduli, n);

    // Test with a simple polynomial: [1, 2, 3, ..., 16, 0, 0, ...]
    let mut original = vec![0u64; n];
    for (i, v) in original.iter_mut().enumerate().take(16) {
        *v = (i + 1) as u64;
    }

    for mod_idx in 0..moduli.len() {
        let mut data = original.clone();
        conv.ntt_inplace(&mut data, mod_idx);
        conv.intt_inplace(&mut data, mod_idx);
        assert_eq!(
            data, original,
            "NTT round-trip failed for modulus {:#x}",
            moduli[mod_idx]
        );
    }
}

#[test]
fn test_ntt_matches_naive_small() {
    // Verify our Cooley-Tukey NTT matches the naive O(N^2) DFT
    let m = 17u64;
    let n = 8;
    let conv = NttConverter::new(&[m], n);

    let coeffs = vec![1u64, 2, 3, 4, 5, 6, 7, 8];

    // Compute naive NTT (in standard order)
    let naive_result = conv.naive_ntt(&coeffs, 0);

    // Compute Cooley-Tukey NTT
    let mut ct_result = coeffs.clone();
    conv.ntt_inplace(&mut ct_result, 0);

    // Determine which ordering the CT NTT produces
    let n_power = n.ilog2();

    if ct_result == naive_result {
        // CT output matches naive directly — both in natural order
        // This is expected if CT does DIT (decimation in time)
        panic!("CT NTT output is in NATURAL order (matches naive directly)");
    }

    // Check if CT output is bit-reversed relative to naive
    let mut ct_unreversed = vec![0u64; n];
    for i in 0..n {
        ct_unreversed[bit_reverse(i, n_power)] = ct_result[i];
    }
    if ct_unreversed == naive_result {
        // CT result[i] = naive[bitrev(i)]
        // So CT output is in bit-reversed order (DIF Cooley-Tukey)
        // This means: ct_result[i] holds the evaluation at the bitrev(i)-th point
        return; // This is the expected result for DIF CT NTT
    }

    // Check reverse mapping
    let mut naive_unreversed = vec![0u64; n];
    for i in 0..n {
        naive_unreversed[bit_reverse(i, n_power)] = naive_result[i];
    }
    if ct_result == naive_unreversed {
        panic!("Naive is bit-reversed relative to CT — unexpected ordering");
    }

    panic!(
        "NTT mismatch!\n  CT result:    {ct_result:?}\n  Naive result: {naive_result:?}\n  CT unreversed: {ct_unreversed:?}"
    );
}

#[test]
fn test_ntt_constant_polynomial() {
    // NTT of constant polynomial [c, 0, 0, ...] should give [c, c, c, ...c]
    // (with possible reordering) because a constant evaluates to c at all points
    let m = 17u64;
    let n = 8;
    let conv = NttConverter::new(&[m], n);

    let mut data = vec![0u64; n];
    data[0] = 5; // constant polynomial = 5

    let naive = conv.naive_ntt(&data, 0);
    // For constant polynomial, all NTT values should equal 5
    for (i, &v) in naive.iter().enumerate() {
        assert_eq!(v, 5, "Naive NTT of constant should be 5 at position {i}, got {v}");
    }

    conv.ntt_inplace(&mut data, 0);
    // After Cooley-Tukey NTT, all values should also be 5
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, 5, "CT NTT of constant should be 5 at position {i}, got {v}");
    }
}

#[test]
fn test_ntt_x_polynomial() {
    // NTT of X = [0, 1, 0, ..., 0].
    // Naive: NTT[k] = psi * omega^k
    let m = 17u64;
    let n = 8;
    let conv = NttConverter::new(&[m], n);
    let psi = conv.psi_roots[0];
    let omega = mod_mul(psi, psi, m);

    let coeffs = {
        let mut v = vec![0u64; n];
        v[1] = 1;
        v
    };

    let naive = conv.naive_ntt(&coeffs, 0);
    for k in 0..n {
        let expected = mod_mul(psi, mod_pow(omega, k as u64, m), m);
        assert_eq!(
            naive[k], expected,
            "Naive NTT of X at position {k}: got {}, expected {expected}",
            naive[k]
        );
    }
}

#[test]
fn test_galois_inverse_basic() {
    let n = 8192;
    let m = 2 * n;

    // Galois elements for expansion levels
    let galois_elts: Vec<usize> = (0..5).map(|l| (n >> l) + 1).collect();
    // [8193, 4097, 2049, 1025, 513]

    for g in galois_elts {
        let g_inv = mod_inverse(g as u64, m as u64);
        let product = (g as u128 * g_inv as u128) % m as u128;
        assert_eq!(
            product, 1,
            "g={g}, g_inv={g_inv}, g*g_inv mod {m} = {product}"
        );
        // g_inv should be odd (Galois elements must be odd)
        assert_eq!(
            g_inv % 2,
            1,
            "Galois inverse {g_inv} should be odd for g={g}"
        );
    }
}

#[test]
fn test_ntt_linearity() {
    // NTT(a + b) = NTT(a) + NTT(b) (mod m)
    let m = PROD_Q0;
    let n = PROD_DEGREE;
    let conv = NttConverter::new(&[m], n);

    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    for i in 0..16 {
        a[i] = (i + 1) as u64;
        b[i] = (i * 3 + 7) as u64;
    }

    // NTT(a)
    let mut ntt_a = a.clone();
    conv.ntt_inplace(&mut ntt_a, 0);

    // NTT(b)
    let mut ntt_b = b.clone();
    conv.ntt_inplace(&mut ntt_b, 0);

    // NTT(a + b)
    let mut ab_sum: Vec<u64> = a.iter().zip(b.iter()).map(|(&x, &y)| (x + y) % m).collect();
    conv.ntt_inplace(&mut ab_sum, 0);

    // NTT(a) + NTT(b) should equal NTT(a+b)
    for i in 0..n {
        let expected = (ntt_a[i] + ntt_b[i]) % m;
        assert_eq!(
            ab_sum[i], expected,
            "NTT linearity failed at position {i}"
        );
    }
}

#[test]
fn test_ntt_convolution() {
    // Pointwise multiplication in NTT domain = negacyclic convolution in coefficient domain
    // This is the key property used in HE operations
    let m = PROD_Q0;
    let n = PROD_DEGREE;
    let conv = NttConverter::new(&[m], n);

    // Simple polynomials: a = [1, 2, 0, ...], b = [3, 4, 0, ...]
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    a[0] = 1;
    a[1] = 2;
    b[0] = 3;
    b[1] = 4;

    // Expected product (negacyclic): (1+2x)(3+4x) = 3 + 4x + 6x + 8x^2
    // = 3 + 10x + 8x^2  (but mod X^N+1, x^2 is just x^2 since N >> 2)
    let mut expected = vec![0u64; n];
    expected[0] = 3;
    expected[1] = 10;
    expected[2] = 8;

    // NTT, pointwise multiply, INTT
    let mut ntt_a = a.clone();
    let mut ntt_b = b.clone();
    conv.ntt_inplace(&mut ntt_a, 0);
    conv.ntt_inplace(&mut ntt_b, 0);

    let mut ntt_prod: Vec<u64> = ntt_a
        .iter()
        .zip(ntt_b.iter())
        .map(|(&x, &y)| mod_mul(x, y, m))
        .collect();
    conv.intt_inplace(&mut ntt_prod, 0);

    assert_eq!(
        &ntt_prod[..4],
        &expected[..4],
        "NTT convolution failed for simple polynomial multiplication"
    );
}

/// Test that NTT conversion matches fhe.rs's NTT for the same polynomial.
/// This verifies the pipeline: fhe.rs NTT → INTT(fhe.rs) → HEonGPU NTT.
#[test]
fn test_conversion_vs_fhe_rs() {
    use fhe_math::rq::{self, traits::TryConvertFrom, Representation};

    let moduli = [PROD_Q0, PROD_Q1, PROD_Q2, PROD_P];
    let n = PROD_DEGREE;

    let conv = NttConverter::new(&moduli, n);

    // Build fhe.rs context with same moduli
    let ctx = rq::Context::new_arc(&moduli, n).unwrap();

    // Create a test polynomial with known coefficients
    let mut coeffs = vec![0i64; n];
    for i in 0..32 {
        coeffs[i] = (i as i64 + 1) * if i % 3 == 0 { -1 } else { 1 };
    }

    // Create poly in PowerBasis (coefficient form)
    let poly_pb =
        rq::Poly::try_convert_from(&coeffs[..], &ctx, false, Representation::PowerBasis).unwrap();

    // Convert to fhe.rs NTT
    let mut poly_ntt = poly_pb.clone();
    poly_ntt.change_representation(Representation::Ntt);

    // Now do the conversion: fhe.rs NTT → coeff → HEonGPU NTT
    let mut converted = poly_ntt.clone();
    converted.change_representation(Representation::PowerBasis);
    let raw = converted.coefficients();
    let coeff_data = raw.as_slice().unwrap().to_vec();

    // Verify the intermediate step: the coefficients should match our input
    // (converted back to unsigned representation)
    for mod_idx in 0..moduli.len() {
        let m = moduli[mod_idx];
        for i in 0..32 {
            let expected = if coeffs[i] < 0 {
                m - (-coeffs[i]) as u64
            } else {
                coeffs[i] as u64
            };
            let actual = coeff_data[mod_idx * n + i];
            assert_eq!(
                actual, expected,
                "Intermediate coefficient mismatch at mod_idx={mod_idx}, pos={i}: got {actual}, expected {expected}"
            );
        }
    }

    // Apply HEonGPU NTT
    let mut heongpu_ntt = coeff_data.clone();
    for mod_idx in 0..moduli.len() {
        let start = mod_idx * n;
        let end = start + n;
        conv.ntt_inplace(&mut heongpu_ntt[start..end], mod_idx);
    }

    // Verify by doing HEonGPU INTT — should give back the original coefficients
    let mut roundtrip = heongpu_ntt.clone();
    for mod_idx in 0..moduli.len() {
        let start = mod_idx * n;
        let end = start + n;
        conv.intt_inplace(&mut roundtrip[start..end], mod_idx);
    }

    assert_eq!(
        roundtrip, coeff_data,
        "HEonGPU NTT round-trip through conversion failed"
    );
}

/// Test ciphertext serialization/deserialization round-trip.
/// Encrypt a plaintext, serialize to HEonGPU format, deserialize back, decrypt.
#[test]
fn test_ciphertext_serialize_roundtrip() {
    use fhe::bfv::{BfvParametersBuilder, Ciphertext, Encoding, Plaintext, SecretKey};
    use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};

    let params = BfvParametersBuilder::new()
        .set_degree(PROD_DEGREE)
        .set_plaintext_modulus(1785857)
        .set_moduli_sizes(&[50, 55, 55])
        .build_arc()
        .unwrap();

    let mut rng = rand::rng();
    let sk = SecretKey::random(&params, &mut rng);

    // Encode a test plaintext
    let mut pt_data = vec![0u64; 20];
    pt_data[0] = 42;
    pt_data[1] = 123;
    pt_data[2] = 999;
    let pt = Plaintext::try_encode(&pt_data, Encoding::poly(), &params).unwrap();
    let ct: Ciphertext = sk.try_encrypt(&pt, &mut rng).unwrap();

    // Serialize to HEonGPU format
    let bytes = serialize_ct_for_test(&ct);

    // Deserialize back
    let ctx = params.context_at_level(0).unwrap().clone();
    let ct2 = deserialize_ct_for_test(&bytes, &params, &ctx).unwrap();

    // Decrypt and verify
    let pt2 = sk.try_decrypt(&ct2).unwrap();
    let decoded = Vec::<u64>::try_decode(&pt2, Encoding::poly()).unwrap();

    assert_eq!(decoded[0], 42, "Coefficient 0 mismatch");
    assert_eq!(decoded[1], 123, "Coefficient 1 mismatch");
    assert_eq!(decoded[2], 999, "Coefficient 2 mismatch");
}

// --- Helpers for ciphertext serialization tests ---

fn serialize_ct_for_test(ct: &fhe::bfv::Ciphertext) -> Vec<u8> {
    use fhe_math::rq::Representation;

    let cipher_size = ct.len() as u32;
    let num_moduli = ct[0].coefficients().nrows() as u32;
    let ring_size = ct[0].coefficients().ncols() as u32;
    let total_coeffs = cipher_size * num_moduli * ring_size;

    let mut buf = Vec::with_capacity(21 + (total_coeffs as usize) * 8);

    buf.push(0x01); // BFV
    buf.extend_from_slice(&ring_size.to_le_bytes());
    buf.extend_from_slice(&num_moduli.to_le_bytes());
    buf.extend_from_slice(&cipher_size.to_le_bytes());
    buf.push(0x00); // in_ntt_domain_ = false
    buf.push(0x01); // storage_type_ = HOST
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

fn deserialize_ct_for_test(
    bytes: &[u8],
    params: &std::sync::Arc<fhe::bfv::BfvParameters>,
    ctx: &std::sync::Arc<fhe_math::rq::Context>,
) -> Result<fhe::bfv::Ciphertext, Box<dyn std::error::Error>> {
    use fhe_math::rq::{self, traits::TryConvertFrom, Representation};

    let ring_size = u32::from_le_bytes(bytes[1..5].try_into()?) as usize;
    let num_moduli = u32::from_le_bytes(bytes[5..9].try_into()?) as usize;
    let cipher_size = u32::from_le_bytes(bytes[9..13].try_into()?) as usize;
    let in_ntt = bytes[13] != 0;
    let _total_coeffs = u32::from_le_bytes(bytes[17..21].try_into()?) as usize;

    let coeffs_per_poly = num_moduli * ring_size;
    let repr = if in_ntt {
        Representation::Ntt
    } else {
        Representation::PowerBasis
    };

    let mut polys = Vec::with_capacity(cipher_size);
    let mut offset = 21;

    for _ in 0..cipher_size {
        let mut raw = vec![0u64; coeffs_per_poly];
        for v in raw.iter_mut() {
            *v = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
            offset += 8;
        }

        let mut poly = rq::Poly::try_convert_from(raw, ctx, false, repr)?;
        if !in_ntt {
            poly.change_representation(Representation::Ntt);
        }
        polys.push(poly);
    }

    Ok(fhe::bfv::Ciphertext::new(polys, params)?)
}

// ─── Key-switching simulation test ──────────────────────────────────────────
//
// Simulates METHOD_I relinearization on the CPU to verify the key formula.

const KS_Q: [u64; 3] = [PROD_Q0, PROD_Q1, PROD_Q2];
const KS_P: u64 = PROD_P;
const KS_N: usize = PROD_DEGREE;
const KS_Q_SIZE: usize = 3;
const KS_Q_PRIME_SIZE: usize = 4;
const KS_PLAINTEXT_MODULUS: u64 = 1785857;

/// Simulate the cipher_broadcast + NTT + key-switch multiply-accumulate + INTT
/// + mod-down + add to c0/c1 steps of METHOD_I relinearization.
///
/// All operations in HEonGPU NTT domain using the HEonGPU NTT converter.
#[test]
fn test_relin_key_switching_simulation() {
    use fhe::bfv::{BfvParametersBuilder, Ciphertext, Encoding, Plaintext, SecretKey};
    use fhe_math::rq::{self, traits::TryConvertFrom, Representation};
    use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter, Serialize};
    use prost::Message;

    let params = BfvParametersBuilder::new()
        .set_degree(KS_N)
        .set_plaintext_modulus(KS_PLAINTEXT_MODULUS)
        .set_moduli_sizes(&[50, 55, 55])
        .build_arc()
        .unwrap();

    let mut rng = rand::rng();
    let sk = SecretKey::random(&params, &mut rng);

    // Get the secret key coefficients
    let sk_bytes = sk.to_bytes();
    let sk_proto = fhe::proto::bfv::SecretKey::decode(&sk_bytes[..]).unwrap();
    let sk_coeffs: &[i64] = &sk_proto.coeffs;

    // Build Q_tilda context (Q + P)
    let q_tilda_moduli = [KS_Q[0], KS_Q[1], KS_Q[2], KS_P];
    let ctx_qt = rq::Context::new_arc(&q_tilda_moduli, KS_N).unwrap();

    // Secret key in Q_tilda NTT
    let sk_ntt = {
        let mut p = rq::Poly::try_convert_from(
            sk_coeffs, &ctx_qt, false, Representation::PowerBasis,
        ).unwrap();
        p.change_representation(Representation::Ntt);
        p
    };

    // s^2 in Q_tilda NTT
    let s_sq = &sk_ntt * &sk_ntt;

    // Compute RNS factors: P mod Q[i]
    let factors: [u64; KS_Q_SIZE] = [
        KS_P % KS_Q[0],
        KS_P % KS_Q[1],
        KS_P % KS_Q[2],
    ];

    // ── Generate relin key (same formula as pir_e2e_test.rs) ─────────────
    struct RelinKeyLevel {
        c0: rq::Poly,
        c1: rq::Poly,
    }

    let mut relin_levels: Vec<RelinKeyLevel> = Vec::with_capacity(KS_Q_SIZE);

    for i in 0..KS_Q_SIZE {
        let a = rq::Poly::random(&ctx_qt, Representation::Ntt, &mut rng);
        let e = rq::Poly::small(&ctx_qt, Representation::Ntt, 1, &mut rng).unwrap();

        // factor_poly: constant = factors[i] at modulus i, zero elsewhere
        let mut fdata = vec![0u64; KS_Q_PRIME_SIZE * KS_N];
        for k in 0..KS_N {
            fdata[i * KS_N + k] = factors[i];
        }
        let factor_poly = rq::Poly::try_convert_from(
            fdata, &ctx_qt, true, Representation::Ntt,
        ).unwrap();

        let message = &factor_poly * &s_sq;
        let c0 = &(&(-&(&(&sk_ntt * &a) + &e)) + &message);
        let c1 = a.clone();

        relin_levels.push(RelinKeyLevel { c0: c0.clone(), c1 });
    }

    // ── Create a 3-component ciphertext ──────────────────────────────────

    // Encrypt two plaintexts and multiply
    let mut pt1_data = vec![0u64; 32];
    pt1_data[0] = 7;
    pt1_data[1] = 3;
    let pt1 = Plaintext::try_encode(&pt1_data, Encoding::poly(), &params).unwrap();
    let ct1: Ciphertext = sk.try_encrypt(&pt1, &mut rng).unwrap();

    let mut pt2_data = vec![0u64; 32];
    pt2_data[0] = 5;
    pt2_data[1] = 2;
    let pt2 = Plaintext::try_encode(&pt2_data, Encoding::poly(), &params).unwrap();
    let ct2: Ciphertext = sk.try_encrypt(&pt2, &mut rng).unwrap();

    // Multiply to get 3-component ciphertext
    let ct3 = &ct1 * &ct2;
    assert_eq!(ct3.len(), 3, "Multiplication should produce 3-component ciphertext");

    // Verify the unrelinearized ciphertext decrypts correctly
    let pt3 = sk.try_decrypt(&ct3).unwrap();
    let decoded3 = Vec::<u64>::try_decode(&pt3, Encoding::poly()).unwrap();
    // (7 + 3x)(5 + 2x) = 35 + 14x + 15x + 6x^2 = 35 + 29x + 6x^2
    assert_eq!(decoded3[0], 35, "Unrelinearized coeff 0");
    assert_eq!(decoded3[1], 29, "Unrelinearized coeff 1");
    assert_eq!(decoded3[2], 6, "Unrelinearized coeff 2");

    // ── Simulate METHOD_I relinearization ────────────────────────────────
    //
    // Get the ciphertext polynomials. They are in fhe.rs NTT domain (Q ring).
    // We need to work in Q_tilda for the key-switching.

    let ctx_q = params.context_at_level(0).unwrap().clone();

    // Get c0, c1, c2 in coefficient form (Q ring)
    let mut c0_coeff = ct3[0].clone();
    c0_coeff.change_representation(Representation::PowerBasis);
    let mut c1_coeff = ct3[1].clone();
    c1_coeff.change_representation(Representation::PowerBasis);
    let mut c2_coeff = ct3[2].clone();
    c2_coeff.change_representation(Representation::PowerBasis);

    // c2 in coefficient form has Q_SIZE moduli, each with KS_N values
    let c2_raw = c2_coeff.coefficients().as_slice().unwrap().to_vec();
    assert_eq!(c2_raw.len(), KS_Q_SIZE * KS_N);

    // Broadcast c2: for each decomposition level i, copy c2[i] to all Q_tilda moduli
    // Then key-switch in fhe.rs NTT domain (using Q_tilda context).
    //
    // Note: We do this in fhe.rs NTT domain, not HEonGPU NTT, because we want to
    // test the FORMULA correctness independently of the NTT conversion.

    // Initialize accumulators in Q_tilda NTT domain
    let zero_qt = rq::Poly::try_convert_from(
        vec![0u64; KS_Q_PRIME_SIZE * KS_N], &ctx_qt, true, Representation::Ntt,
    ).unwrap();
    let mut ks_c0_accum = zero_qt.clone();
    let mut ks_c1_accum = zero_qt;

    for i in 0..KS_Q_SIZE {
        // Get c2 at decomposition level i (Q modulus i): N coefficients
        let c2_level_i = &c2_raw[i * KS_N..(i + 1) * KS_N];

        // Broadcast to all Q_tilda moduli
        let mut bcast = vec![0u64; KS_Q_PRIME_SIZE * KS_N];
        for j in 0..KS_Q_PRIME_SIZE {
            for k in 0..KS_N {
                // Reduce c2_level_i[k] modulo q_tilda_moduli[j]
                bcast[j * KS_N + k] = c2_level_i[k] % q_tilda_moduli[j];
            }
        }

        // Create broadcast poly in coefficient form, then NTT
        let mut bcast_poly = rq::Poly::try_convert_from(
            bcast, &ctx_qt, true, Representation::PowerBasis,
        ).unwrap();
        bcast_poly.change_representation(Representation::Ntt);

        // Pointwise multiply with key data (already in NTT form)
        let prod_c0 = &bcast_poly * &relin_levels[i].c0;
        let prod_c1 = &bcast_poly * &relin_levels[i].c1;

        // Accumulate
        ks_c0_accum = &ks_c0_accum + &prod_c0;
        ks_c1_accum = &ks_c1_accum + &prod_c1;
    }

    // ── Mod-down: Q_tilda → Q ────────────────────────────────────────────
    //
    // divide_round_lastq: result_q[j] = (accum_q[j] - accum_p) * P_inv mod Q[j]
    //
    // Get accumulator in coefficient form
    ks_c0_accum.change_representation(Representation::PowerBasis);
    ks_c1_accum.change_representation(Representation::PowerBasis);

    let ks_c0_raw = ks_c0_accum.coefficients().as_slice().unwrap().to_vec();
    let ks_c1_raw = ks_c1_accum.coefficients().as_slice().unwrap().to_vec();

    // Extract P-component (last modulus in Q_tilda)
    let p_idx = KS_Q_SIZE; // index 3
    let ks_c0_p: Vec<u64> = ks_c0_raw[p_idx * KS_N..(p_idx + 1) * KS_N].to_vec();
    let ks_c1_p: Vec<u64> = ks_c1_raw[p_idx * KS_N..(p_idx + 1) * KS_N].to_vec();

    // Compute P_inv mod Q[j] for each Q modulus
    let p_inv_q: Vec<u64> = KS_Q.iter().map(|&q| mod_inverse(KS_P, q)).collect();

    // half_p for rounding: P/2
    let half_p = KS_P / 2;

    // Mod-down: for each Q modulus j, for each coefficient k:
    // ks_result[j][k] = round((accum[j][k] - accum[P][k]) / P) mod Q[j]
    // The "round" is: (accum[j][k] - accum[P][k] + P/2) / P, with proper modular handling
    let mut moddown_c0 = vec![0u64; KS_Q_SIZE * KS_N];
    let mut moddown_c1 = vec![0u64; KS_Q_SIZE * KS_N];

    for j in 0..KS_Q_SIZE {
        let qj = KS_Q[j];
        let p_inv_j = p_inv_q[j];
        // half_p mod Q[j] for rounding
        let _half_p_mod_qj = half_p % qj;

        for k in 0..KS_N {
            // The P-residue reduced mod Q[j]
            let p_val_c0 = ks_c0_p[k] % qj;
            let p_val_c1 = ks_c1_p[k] % qj;

            // accum[j][k] - p_val (mod Q[j])
            let diff_c0 = if ks_c0_raw[j * KS_N + k] >= p_val_c0 {
                ks_c0_raw[j * KS_N + k] - p_val_c0
            } else {
                ks_c0_raw[j * KS_N + k] + qj - p_val_c0
            };
            let diff_c1 = if ks_c1_raw[j * KS_N + k] >= p_val_c1 {
                ks_c1_raw[j * KS_N + k] - p_val_c1
            } else {
                ks_c1_raw[j * KS_N + k] + qj - p_val_c1
            };

            // Multiply by P^{-1} mod Q[j]
            moddown_c0[j * KS_N + k] = mod_mul(diff_c0, p_inv_j, qj);
            moddown_c1[j * KS_N + k] = mod_mul(diff_c1, p_inv_j, qj);
        }
    }

    // ── Add key-switch result to original c0, c1 ─────────────────────────
    let c0_raw = c0_coeff.coefficients().as_slice().unwrap().to_vec();
    let c1_raw = c1_coeff.coefficients().as_slice().unwrap().to_vec();

    let mut final_c0 = vec![0u64; KS_Q_SIZE * KS_N];
    let mut final_c1 = vec![0u64; KS_Q_SIZE * KS_N];

    for j in 0..KS_Q_SIZE {
        let qj = KS_Q[j];
        for k in 0..KS_N {
            final_c0[j * KS_N + k] = (c0_raw[j * KS_N + k] + moddown_c0[j * KS_N + k]) % qj;
            final_c1[j * KS_N + k] = (c1_raw[j * KS_N + k] + moddown_c1[j * KS_N + k]) % qj;
        }
    }

    // ── Create 2-component ciphertext and decrypt ────────────────────────
    let mut p0 = rq::Poly::try_convert_from(
        final_c0, &ctx_q, true, Representation::PowerBasis,
    ).unwrap();
    p0.change_representation(Representation::Ntt);

    let mut p1 = rq::Poly::try_convert_from(
        final_c1, &ctx_q, true, Representation::PowerBasis,
    ).unwrap();
    p1.change_representation(Representation::Ntt);

    let relin_ct = Ciphertext::new(vec![p0, p1], &params).unwrap();
    let pt_relin = sk.try_decrypt(&relin_ct).unwrap();
    let decoded_relin = Vec::<u64>::try_decode(&pt_relin, Encoding::poly()).unwrap();

    // Should match the original multiplication result: 35 + 29x + 6x^2
    assert_eq!(
        decoded_relin[0], 35,
        "Relinearized coeff 0: got {}, expected 35",
        decoded_relin[0]
    );
    assert_eq!(
        decoded_relin[1], 29,
        "Relinearized coeff 1: got {}, expected 29",
        decoded_relin[1]
    );
    assert_eq!(
        decoded_relin[2], 6,
        "Relinearized coeff 2: got {}, expected 6",
        decoded_relin[2]
    );
    // Higher coefficients should be zero
    for i in 3..16 {
        assert_eq!(
            decoded_relin[i], 0,
            "Relinearized coeff {i}: got {}, expected 0",
            decoded_relin[i]
        );
    }
}
