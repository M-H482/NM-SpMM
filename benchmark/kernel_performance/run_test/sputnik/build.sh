source env.sh

SPUTNIK_ROOT=/public/home/macong/mybuild/sputnik
CUDA_ROOT=/public/software/compiler/cuda-12.2
NVCC=${CUDA_ROOT}/bin/nvcc

CUDA_ARCH="-gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89"

${NVCC} -forward-unknown-to-host-compiler -I${CUDA_ROOT}/include -I${SPUTNIK_ROOT} -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L${CUDA_ROOT}/lib64  -L${SPUTNIK_ROOT}/build/sputnik -lcudart -lspmm $CUDA_ARCH -std=c++14  test_sputnik.cu -o test_sputnik
