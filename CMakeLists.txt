cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NM_SpMM CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# Set CUDA properties
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 -arch=sm_80 -Xcompiler=-fopenmp")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Add source files for kernels and preprocessing
set(SOURCES
    src/kernel_32x32_4x4.cu
    src/kernel_32x64_8x4.cu
    src/kernel_64x64_8x8.cu
    src/kernel_64x128_8x8.cu
    src/preprocessing.cu
)

# Add executable for low sparsity test
cuda_add_executable(test_nmspmm
    ${SOURCES}
    tests/test_nmspmm.cu
)

# Link libraries
target_link_libraries(test_nmspmm
    ${OpenMP_CXX_LIBRARIES}
    ${CUDA_LIBRARIES}
)
