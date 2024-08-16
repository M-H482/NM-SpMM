module load cuda/cuda_12.2/12.2

CUDA_ARCH="-gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89"

nvcc --generate-line-info -std=c++11 -O3 $CUDA_ARCH -Xcompiler -fopenmp -lcublas nmsparse.cu -o nmsparse
