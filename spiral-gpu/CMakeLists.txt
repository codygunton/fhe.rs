cmake_minimum_required(VERSION 3.26)
project(spiral-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture: 120 = RTX 5090 (Blackwell); also allow override via cmake -DCMAKE_CUDA_ARCHITECTURES=75
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures to compile for")

# Find CUDAToolkit before HEonGPU (HEonGPU needs it for Thrust/CCCL detection)
find_package(CUDAToolkit REQUIRED)

# HEonGPU — only needed for GPU-NTT butterfly primitives and RMM
set(HEONGPU_ROOT "$ENV{HOME}/HEonGPU" CACHE PATH "Path to HEonGPU source tree")
set(HEONGPU_CUDA_ARCH_FORCE_MANUAL ON CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory(${HEONGPU_ROOT} ${CMAKE_BINARY_DIR}/heongpu EXCLUDE_FROM_ALL)

# Third-party: cpp-httplib (header-only) and nlohmann/json
include(FetchContent)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.1
)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(httplib nlohmann_json)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

add_subdirectory(server)

message(STATUS "")
message(STATUS "=== Spiral GPU Server Configuration ===")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "HEonGPU root:       ${HEONGPU_ROOT}")
message(STATUS "")
