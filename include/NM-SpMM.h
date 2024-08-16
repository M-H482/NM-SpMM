#ifndef NMSpMM_H
#define NMSpMM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define VEC_LEN (32)

void PreProcessing_low_sparsity(int* DT, int W, int Q, int Ns);
void PreProcessing_high_sparsity(int* column_info, int pruning_M, int pruning_N, int* DT, int W, int Q, int N, int Ns);
void transIndex(int* DT, int W, int N, int Q, int& pruning_N, int& pruning_M);

void nmGEMM_huge_matrices_low_sparsity(float* A, float* B, int* D, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);
void nmGEMM_huge_matrices_high_sparsity(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);

void nmGEMM_large_matrices_low_sparsity(float* A, float* B, int* D, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);
void nmGEMM_large_matrices_high_sparsity(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);

void nmGEMM_medium_matrices_low_sparsity(float* A, float* B, int* D, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);
void nmGEMM_medium_matrices_high_sparsity(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);

void nmGEMM_small_matrices_low_sparsity(float* A, float* B, int* D, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);
void nmGEMM_small_matrices_high_sparsity(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K);

#endif // NMSpMM_H
