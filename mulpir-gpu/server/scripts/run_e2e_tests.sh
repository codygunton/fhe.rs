#!/usr/bin/env bash
#
# Run end-to-end compatibility tests between fhe.rs and the MulPIR GPU server.
# This generates test vectors from fhe.rs, builds the C++ tests, and runs them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
TEST_VECTORS_DIR="${PROJECT_DIR}/test_vectors"

echo "=== MulPIR E2E Tests ==="
echo ""

# Step 1: Generate test vectors from fhe.rs
if [ ! -d "${TEST_VECTORS_DIR}" ] || [ "${FORCE_REGEN:-0}" -eq 1 ]; then
    echo "Step 1: Generating test vectors from fhe.rs..."
    "${SCRIPT_DIR}/generate_test_vectors.sh"
    echo ""
else
    echo "Step 1: Test vectors already exist at ${TEST_VECTORS_DIR} (use FORCE_REGEN=1 to regenerate)"
    echo ""
fi

# Step 2: Build C++ tests
echo "Step 2: Building GPU tests..."
if [ ! -d "${BUILD_DIR}" ]; then
    cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DMULPIR_BUILD_TESTS=ON
fi
cmake --build "${BUILD_DIR}" --target mulpir_tests --parallel
echo ""

# Step 3: Run compatibility tests
echo "Step 3: Running compatibility tests..."
cd "${BUILD_DIR}"
./mulpir_tests --gtest_filter='Compat*'

echo ""
echo "=== All E2E tests passed! ==="
