#ifndef UTILS_H
#define UTILS_H

#include "omp.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

void init_data(float* A, float* B, int* D, float* BT, int* DT, float* C, int M, int N, int K, int pruning_M, float sparsity)
{
    /**
     *  A:      col-major
     *  B:      col-major
     *  D:      col-major
     *  BT:     row-major
     *  DT:     row-major
     */

    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));

    const unsigned int W = K * (1.0f - sparsity);
    const unsigned int pruning_N = pruning_M * (1.0f - sparsity);
    const unsigned int Q = (int)(N / VEC_LEN);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i + j * M] = 0.0f;
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A[i + j * M] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < W; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i + j * W] = (float)rand() / RAND_MAX;
            BT[i * N + j] = B[i + j * W];
        }
    }

    int* tmp_index = (int*)malloc(sizeof(int) * pruning_M);
    for (int i = 0; i < pruning_M; ++i)
        tmp_index[i] = i;

    std::mt19937 gen(std::random_device {}());
    for (int j = 0; j < N; j += VEC_LEN) {
        int a = (int)(j / VEC_LEN);
        for (int k = 0; k < W; k += pruning_N) {

            std::shuffle(tmp_index, tmp_index + pruning_M, gen);
            std::sort(tmp_index, tmp_index + pruning_N);

            for (int u = 0; u < pruning_N; ++u) {
                D[(k + u) + a * W] = tmp_index[u] + k / (1.0f - sparsity);
                DT[(k + u) * Q + a] = tmp_index[u];
            }
        }
    }
    free(tmp_index);
}

bool allclose(float* A, float* B, int n)
{
    // absolute(a - b) <= (atol + rtol * absolute(b))
    for (int i = 0; i < n; i++) {
        float a = A[i], b = B[i];
        float rtol = 1e-5, atol = 1e-8;
        if (!(fabs(a - b) <= (atol + rtol * fabs(b)))) {
            printf("Error on index %d, (%f, %f)\n", i, a, b);
            return false;
        }
    }
    return true;
}

void nmGEMM_on_cpu(float* A, float* B, int* D, float* C, int M, int N, int K, int W)
{
    int num_threads = omp_get_max_threads();
    printf("Using %d threads compute reference on CPU\n", num_threads);
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < W; k++) {
            for (int j = 0; j < N; j++) {
                int a = (int)(j / VEC_LEN);
                C[i * N + j] += A[i + D[k + a * W] * M] * B[k + j * W];
            }
        }
    }
}

#endif // UTILS_H