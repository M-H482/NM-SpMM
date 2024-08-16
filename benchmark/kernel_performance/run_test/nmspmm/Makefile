# Compiler and CUDA options
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -Xcompiler -fopenmp --generate-line-info
CUDA_ARCH = -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89

INCLUDES = -I$(NM_SpMM_HOME)/include

NM_SpMM_HOME = ../../../..

# Source files
SRC_FILES = \
    $(NM_SpMM_HOME)/src/kernel_64x128_8x8.cu \
    $(NM_SpMM_HOME)/src/kernel_64x64_8x8.cu \
    $(NM_SpMM_HOME)/src/kernel_32x64_8x4.cu \
    $(NM_SpMM_HOME)/src/kernel_32x32_4x4.cu \
    $(NM_SpMM_HOME)/src/preprocessing.cu

# Output directory
BUILD_DIR = $(NM_SpMM_HOME)/benchmark/kernel_performance/run_test/nmspmm

# Targets
all: $(BUILD_DIR)/test_nmspmm

$(BUILD_DIR)/test_nmspmm: $(BUILD_DIR)/test_nmspmm.cu $(SRC_FILES)
	$(NVCC) $(CUDA_ARCH) $(CXXFLAGS) $(INCLUDES) $^ -lcuda -lcublas -o $@

# Clean up build artifacts
clean:
	rm -f $(BUILD_DIR)/test_nmspmm 