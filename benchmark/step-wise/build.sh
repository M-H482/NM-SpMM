set -x

nvcc v0_naive.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v0_naive

nvcc v1_block_tiling.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v1_block_tiling

nvcc v2_warp_thread_tiling.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v2_warp_thread_tiling

nvcc v3_sparsity_aware_low_sparsity.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v3_sparsity_aware_low_sparsity

nvcc v3_sparsity_aware_high_sparsity.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v3_sparsity_aware_high_sparsity

nvcc v4_prefetch_low_sparsity.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v4_prefetch_low_sparsity

nvcc v4_prefetch_high_sparsity.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v4_prefetch_high_sparsity

nvcc v5_splitK_low_sparsity.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v5_splitK_low_sparsity

nvcc v5_splitK_high_sparsity.cu --generate-line-info -std=c++11 -O3 -arch sm_80 -Xcompiler -fopenmp -lcublas -o v5_splitK_high_sparsity
