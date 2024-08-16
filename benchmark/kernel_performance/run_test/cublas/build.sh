module load cuda/cuda_12.2/12.2

CUDA_ARCH="-gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89"

nvcc cublas_f32.cu -std=c++11 -O3 ${CUDA_ARCH} -lcublas -o cublas_f32