# Spiral GPU server — sources and targets

set(SPIRAL_GPU_SOURCES
    src/types_impl.cu
    src/params_impl.cpp
    src/serialization.cpp
    src/kernels/arith.cu
    src/kernels/ntt.cu
    src/kernels/automorph.cu
    src/kernels/gadget.cu
    src/kernels/poly_ops.cu
    src/pipeline/database.cu
    src/pipeline/expand_query.cu
    src/pipeline/db_multiply.cu
    src/pipeline/fold_pack.cu
    src/pipeline/process_query.cu
)

add_library(spiral_gpu STATIC ${SPIRAL_GPU_SOURCES})
target_include_directories(spiral_gpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_include_directories(spiral_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
        ${HEONGPU_ROOT}/thirdparty/GPU-NTT/src/include
        ${CUDAToolkit_INCLUDE_DIRS}/cccl
)
target_link_libraries(spiral_gpu
    PUBLIC
        CUDA::cudart
        heongpu
        Threads::Threads
)
set_target_properties(spiral_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

add_executable(spiral_gpu_server src/main.cpp)
target_include_directories(spiral_gpu_server
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
)
target_link_libraries(spiral_gpu_server
    PRIVATE
        spiral_gpu
        rmm::rmm_logger_impl
        httplib::httplib
        nlohmann_json::nlohmann_json
        OpenSSL::Crypto
)
set_target_properties(spiral_gpu_server PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS spiral_gpu spiral_gpu_server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
