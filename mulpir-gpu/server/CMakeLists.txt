cmake_minimum_required(VERSION 3.26)
project(mulpir-gpu-server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(MULPIR_BUILD_TESTS "Build test suite" ON)
option(MULPIR_BUILD_BENCHMARKS "Build benchmarks" ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures to compile for")

# Find CUDAToolkit first (needed by HEonGPU)
find_package(CUDAToolkit REQUIRED)

# HEonGPU - include as subdirectory for proper transitive dependency handling
set(HEONGPU_ROOT "$ENV{HOME}/HEonGPU" CACHE PATH "Path to HEonGPU source tree")
set(HEONGPU_CUDA_ARCH_FORCE_MANUAL ON CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HEonGPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory(${HEONGPU_ROOT} ${CMAKE_BINARY_DIR}/heongpu EXCLUDE_FROM_ALL)

# Find other required packages
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# Optional: GoogleTest for testing
if(MULPIR_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# Compile definitions
add_compile_definitions(
    $<$<CONFIG:Debug>:MULPIR_DEBUG>
)

# Core library sources
set(MULPIR_SOURCES
    src/encoding/tile_encoder.cu
    src/database/database_manager.cu
    src/pir/query_expander.cu
    src/pir/dot_product.cu
    src/pir/selector.cu
    src/pir/pir_engine.cu
    src/serialization/wire_format.cu
    src/server/pir_server.cpp
)

# Core library
add_library(mulpir_gpu STATIC ${MULPIR_SOURCES})
target_link_libraries(mulpir_gpu
    PUBLIC
        heongpu
        CUDA::cudart
        OpenSSL::Crypto
        Threads::Threads
)
target_include_directories(mulpir_gpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(mulpir_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Server executable
# Link the RMM logger impl source - RMM provides it as INTERFACE sources
# which need to be compiled by the final executable target
add_executable(mulpir_server apps/server_main.cpp)
target_link_libraries(mulpir_server PRIVATE mulpir_gpu rmm::rmm_logger_impl)

# Benchmarks
if(MULPIR_BUILD_BENCHMARKS)
    add_executable(mulpir_benchmark apps/benchmark.cu)
    target_link_libraries(mulpir_benchmark PRIVATE mulpir_gpu rmm::rmm_logger_impl)
    set_target_properties(mulpir_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Tests
if(MULPIR_BUILD_TESTS)
    add_executable(mulpir_tests
        tests/test_encoding.cu
        tests/test_pir_engine.cu
        tests/test_compat.cu
    )
    target_link_libraries(mulpir_tests
        PRIVATE
            mulpir_gpu
            rmm::rmm_logger_impl
            GTest::gtest
            GTest::gtest_main
    )
    target_include_directories(mulpir_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_target_properties(mulpir_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    include(GoogleTest)
    gtest_discover_tests(mulpir_tests)
endif()

# Install targets
install(TARGETS mulpir_gpu mulpir_server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Summary
message(STATUS "")
message(STATUS "=== MulPIR GPU Server Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "HEonGPU root: ${HEONGPU_ROOT}")
message(STATUS "Build tests: ${MULPIR_BUILD_TESTS}")
message(STATUS "Build benchmarks: ${MULPIR_BUILD_BENCHMARKS}")
message(STATUS "")
