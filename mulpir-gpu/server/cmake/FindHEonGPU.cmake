# FindHEonGPU.cmake
# Find the HEonGPU library for GPU-accelerated homomorphic encryption
#
# This module defines:
#   HEONGPU_FOUND        - True if HEonGPU was found
#   HEONGPU_INCLUDE_DIRS - HEonGPU include directories
#   HEONGPU_LIBRARIES    - HEonGPU libraries to link against
#
# Hints:
#   HEONGPU_ROOT         - Root directory of HEonGPU installation
#   HEONGPU_INCLUDE_DIR  - Directory containing HEonGPU headers
#   HEONGPU_LIBRARY_DIR  - Directory containing HEonGPU libraries

# Search for HEonGPU include directory
find_path(HEONGPU_INCLUDE_DIR
    NAMES
        heongpu/heongpu.cuh
        heongpu/HEContext.cuh
    PATHS
        ${HEONGPU_ROOT}
        ${HEONGPU_ROOT}/include
        /usr/local/include
        /usr/include
        $ENV{HOME}/HEonGPU/include
        $ENV{HEONGPU_ROOT}/include
    DOC "HEonGPU include directory"
)

# Search for HEonGPU library
find_library(HEONGPU_LIBRARY
    NAMES
        heongpu
        libheongpu
    PATHS
        ${HEONGPU_ROOT}
        ${HEONGPU_ROOT}/lib
        ${HEONGPU_ROOT}/lib64
        ${HEONGPU_LIBRARY_DIR}
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        $ENV{HOME}/HEonGPU/lib
        $ENV{HEONGPU_ROOT}/lib
    DOC "HEonGPU library"
)

# Handle REQUIRED and QUIET arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HEonGPU
    REQUIRED_VARS
        HEONGPU_INCLUDE_DIR
        HEONGPU_LIBRARY
    FAIL_MESSAGE
        "Could not find HEonGPU. Set HEONGPU_ROOT to the installation directory."
)

# Set output variables
if(HEONGPU_FOUND)
    set(HEONGPU_INCLUDE_DIRS ${HEONGPU_INCLUDE_DIR})
    set(HEONGPU_LIBRARIES ${HEONGPU_LIBRARY})

    # Check for additional dependencies that HEonGPU might need
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        list(APPEND HEONGPU_LIBRARIES CUDA::cudart CUDA::curand)
    endif()

    # Mark as advanced
    mark_as_advanced(
        HEONGPU_INCLUDE_DIR
        HEONGPU_LIBRARY
    )

    # Create imported target if it doesn't exist
    if(NOT TARGET HEonGPU::HEonGPU)
        add_library(HEonGPU::HEonGPU UNKNOWN IMPORTED)
        set_target_properties(HEonGPU::HEonGPU PROPERTIES
            IMPORTED_LOCATION "${HEONGPU_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HEONGPU_INCLUDE_DIR}"
        )
    endif()
endif()
