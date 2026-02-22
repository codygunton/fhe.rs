#!/usr/bin/env bash
#
# Generate test vectors for MulPIR GPU server compatibility tests.
# Requires fhe.rs to be built with the generate_test_vectors example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FHE_RS_DIR="$(dirname "$PROJECT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/test_vectors"

echo "=== MulPIR Test Vector Generator ==="
echo ""
echo "fhe.rs directory: $FHE_RS_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo ""

# Build the test vector generator
echo "Building test vector generator..."
cargo build --manifest-path "${FHE_RS_DIR}/Cargo.toml" \
    --example generate_test_vectors \
    --release \
    -p fhe

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the generator
echo "Generating test vectors..."
cargo run --manifest-path "${FHE_RS_DIR}/Cargo.toml" \
    --example generate_test_vectors \
    --release \
    -p fhe \
    -- \
    --output-dir "$OUTPUT_DIR" \
    --num-tiles 100 \
    --tile-size 20480

echo ""
echo "Test vectors generated at: $OUTPUT_DIR"
echo "Contents:"
ls -lh "$OUTPUT_DIR"
