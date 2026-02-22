#!/bin/bash
# Build and run MulPIR GPU server tests
# This script generates test vectors (if needed) and runs the C++ test suite

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_DIR}/build"
TEST_VECTORS_DIR="${PROJECT_DIR}/test_vectors"

# Parse arguments
BUILD_TYPE=${BUILD_TYPE:-Release}
VERBOSE=${VERBOSE:-0}
SKIP_VECTORS=${SKIP_VECTORS:-0}

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --debug         Build in debug mode"
    echo "  --verbose       Verbose test output"
    echo "  --skip-vectors  Skip test vector generation"
    echo "  --clean         Clean build directory first"
    echo "  --help          Show this help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --skip-vectors)
            SKIP_VECTORS=1
            shift
            ;;
        --clean)
            rm -rf "${BUILD_DIR}"
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "=== MulPIR GPU Server Test Runner ==="
echo "Build type: ${BUILD_TYPE}"
echo ""

# Generate test vectors if needed
if [ "${SKIP_VECTORS}" -eq 0 ] && [ ! -d "${TEST_VECTORS_DIR}" ]; then
    echo "Generating test vectors..."
    "${SCRIPT_DIR}/generate_test_vectors.sh"
    echo ""
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo "Configuring build..."
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DMULPIR_BUILD_TESTS=ON \
      -DMULPIR_BUILD_BENCHMARKS=ON \
      "${PROJECT_DIR}"

# Build
echo ""
echo "Building..."
cmake --build . --parallel "$(nproc)"

# Run tests
echo ""
echo "Running tests..."
if [ "${VERBOSE}" -eq 1 ]; then
    ctest --output-on-failure --verbose
else
    ctest --output-on-failure
fi

echo ""
echo "=== All tests passed! ==="
