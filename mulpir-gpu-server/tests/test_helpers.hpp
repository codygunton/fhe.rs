#pragma once

#include <cstdint>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace mulpir::test {

/// Create a fully configured BFV context matching fhe.rs parameters.
inline HEContextPtr create_test_context() {
    auto context = heongpu::GenHEContext<heongpu::Scheme::BFV>();
    context->set_poly_modulus_degree(BFVConfig::POLY_DEGREE);
    context->set_coeff_modulus_bit_sizes(
        {BFVConfig::MODULI_BITS[0], BFVConfig::MODULI_BITS[1], BFVConfig::MODULI_BITS[2]},
        {BFVConfig::P_MODULI_BITS[0]}
    );
    context->set_plain_modulus(BFVConfig::PLAINTEXT_MODULUS);
    context->generate();
    return context;
}

/// Compute the Galois elements needed for query expansion.
inline std::vector<uint32_t> expansion_galois_elements(size_t expansion_level) {
    std::vector<uint32_t> elts;
    elts.reserve(expansion_level);
    for (size_t l = 0; l < expansion_level; ++l) {
        elts.push_back(static_cast<uint32_t>((BFVConfig::POLY_DEGREE >> l) + 1));
    }
    return elts;
}

/// Key bundle for testing — generates all keys needed for PIR.
struct TestKeys {
    heongpu::Secretkey<heongpu::Scheme::BFV> secret_key;
    heongpu::Publickey<heongpu::Scheme::BFV> public_key;
    heongpu::Relinkey<heongpu::Scheme::BFV> relin_key;
    heongpu::Galoiskey<heongpu::Scheme::BFV> galois_key;

    static TestKeys generate(HEContextPtr context, size_t expansion_level) {
        heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);

        heongpu::Secretkey<heongpu::Scheme::BFV> sk(context);
        keygen.generate_secret_key(sk);

        heongpu::Publickey<heongpu::Scheme::BFV> pk(context);
        keygen.generate_public_key(pk, sk);

        heongpu::Relinkey<heongpu::Scheme::BFV> rk(context);
        keygen.generate_relin_key(rk, sk);

        auto galois_elts = expansion_galois_elements(expansion_level);
        heongpu::Galoiskey<heongpu::Scheme::BFV> gk(context, galois_elts);
        keygen.generate_galois_key(gk, sk);

        return TestKeys{
            std::move(sk),
            std::move(pk),
            std::move(rk),
            std::move(gk),
        };
    }
};

}  // namespace mulpir::test
