#include "NM-SpMM.h"
#include "utils.h"
#include <iostream>

typedef void (*nmGEMMFuncPtr_low_sparsity)(float*, float*, int*, float*, int, int, int, int, float, int);
typedef void (*nmGEMMFuncPtr_high_sparsity)(float*, float*, int*, int*, float*, int, int, int, int, float, int);

void test_performance(nmGEMMFuncPtr_low_sparsity low_sparsity_kernel, nmGEMMFuncPtr_high_sparsity high_sparsity_kernel, int Ns,
    int M, int N, int K,
    int pruning_M, float sparsity, int SPLIT_K,
    int warm_up, int iter)
{
    bool is_high_sparsity = ((fabs(sparsity - 0.75f) < 1e-6) || (fabs(sparsity - 0.875f) < 1e-6));
    // Allocate host memory
    const int W = (int)(K * (1.0f - sparsity));
    const int Q = (int)(N / VEC_LEN);
    int pruning_N = (int)(pruning_M * (1.0f - sparsity));

    const int A_nBytes = sizeof(float) * M * K;
    const int C_nBytes = sizeof(float) * M * N;
    const int B_nBytes = sizeof(float) * W * N;
    const int D_nBytes = sizeof(int) * W * Q;
    const int column_nBytes = ((W / pruning_N) * (N / Ns) + 1) * (pruning_M) * sizeof(int);

    float* hA = (float*)malloc(A_nBytes);
    float* hB = (float*)malloc(B_nBytes);
    float* hB_T = (float*)malloc(B_nBytes);
    float* hC = (float*)malloc(C_nBytes);
    float* deviceRes = (float*)malloc(C_nBytes);
    int* hD = (int*)malloc(D_nBytes);
    int* hD_T = (int*)malloc(D_nBytes);
    int* column_info = (int*)malloc(column_nBytes);

    // Check malloc errors
    if (!hA || !hB || !hB_T || !hC || !deviceRes || !hD || !hD_T || !column_info) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return;
    }

    float *dA, *dB, *dC;
    int *dD, *dcolumn_info;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&dA, A_nBytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for dA!" << std::endl;
        return;
    }
    err = cudaMalloc((void**)&dB, B_nBytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for dB!" << std::endl;
        return;
    }
    err = cudaMalloc((void**)&dC, C_nBytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for dC!" << std::endl;
        return;
    }
    err = cudaMalloc((void**)&dD, D_nBytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for dD!" << std::endl;
        return;
    }
    err = cudaMalloc((void**)&dcolumn_info, column_nBytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for dcolumn_info!" << std::endl;
        return;
    }

    // Initialize data
    init_data(hA, hB, hD, hB_T, hD_T, hC, M, N, K, pruning_M, sparsity);

    // Preprocess data
    if (!is_high_sparsity) {
        PreProcessing_low_sparsity(hD_T, W, Q, Ns);
    } else {
        transIndex(hD_T, W, N, Q, pruning_N, pruning_M);
        PreProcessing_high_sparsity(column_info, pruning_M, pruning_N, hD_T, W, Q, N, Ns);
    }

    // Copy data to device
    cudaMemcpy(dA, hA, A_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_T, B_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, C_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dD, hD_T, D_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dcolumn_info, column_info, column_nBytes, cudaMemcpyHostToDevice);

    // Perform computation on CPU
    nmGEMM_on_cpu(hA, hB, hD, hC, M, N, K, W);

    // Run kernel and synchronize
    if (!is_high_sparsity) {
        low_sparsity_kernel(dA, dB, dD, dC, M, N, K, W, sparsity, SPLIT_K);
    } else {
        high_sparsity_kernel(dA, dB, dD, dcolumn_info, dC, M, N, K, W, sparsity, SPLIT_K);
    }
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(deviceRes, dC, C_nBytes, cudaMemcpyDeviceToHost);

    // Check results
    if (allclose(deviceRes, hC, M * N)) {
        printf("The result is right!\n");
    } else {
        printf("The result is wrong !!!!!!!!!!\n");
    }

    // Measures kernel execution time and computes TFLOPS performance.
    float milliseconds = 0.0f, tflops = -1.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warm_up; i++) {
        if (!is_high_sparsity) {
            low_sparsity_kernel(dA, dB, dD, dC, M, N, K, W, sparsity, SPLIT_K);
        } else {
            high_sparsity_kernel(dA, dB, dD, dcolumn_info, dC, M, N, K, W, sparsity, SPLIT_K);
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < iter; i++) {
        if (!is_high_sparsity) {
            low_sparsity_kernel(dA, dB, dD, dC, M, N, K, W, sparsity, SPLIT_K);
        } else {
            high_sparsity_kernel(dA, dB, dD, dcolumn_info, dC, M, N, K, W, sparsity, SPLIT_K);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds = milliseconds / iter;

    tflops = (2.0f * M * N * K / 1e12) / (milliseconds / 1e3);
    printf("Time elapsed: %f ms, %f TFLOPS\n", milliseconds, tflops);

    // Clean up
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
    cudaFree(dcolumn_info);

    free(hA);
    free(hB);
    free(hB_T);
    free(hC);
    free(deviceRes);
    free(hD);
    free(hD_T);
    free(column_info);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv)
{
    int M = 4096, N = 4096, K = 4096;
    int pruning_M = 32, SPLIT_K = 1;
    float sparsity = 0.5f;
    int warm_up = 100, iter = 100;

    if (argc != 1 && argc != 4 && argc != 6 && argc != 7 && argc != 9) {
        printf("Usage: xxxx\n");
        return -1;
    }

    if (argc != 1) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc == 6) {
        pruning_M = atoi(argv[4]);
        sparsity = atof(argv[5]);
    } else if (argc == 7) {
        pruning_M = atoi(argv[4]);
        sparsity = atof(argv[5]);
        SPLIT_K = atoi(argv[6]);
    } else if (argc == 9) {
        pruning_M = atoi(argv[4]);
        sparsity = atof(argv[5]);
        SPLIT_K = atoi(argv[6]);
        warm_up = atoi(argv[7]);
        iter = atoi(argv[8]);
    }
    printf("M = %d, N = %d, K = %d, pruning_M = %d, sparsity = %f, SPLIT_K = %d\n", M, N, K, pruning_M, sparsity, SPLIT_K);

    if (!(fabs(sparsity - 0.0f) < 1e-6 || fabs(sparsity - 0.5f) < 1e-6 || fabs(sparsity - 0.625f) < 1e-6 || fabs(sparsity - 0.75f) < 1e-6 || fabs(sparsity - 0.875f) < 1e-6)) {
        printf("sparsity not in {0.5, 0.625, 0.75, 0.875}!!!\n");
        return -2;
    }

    if (K % pruning_M != 0) {
        printf("K is not an integer multiple of pruning_M!!!\n");
        return -3;
    }
    nmGEMMFuncPtr_low_sparsity low_sparsity_kernel = nullptr;
    nmGEMMFuncPtr_high_sparsity high_sparsity_kernel = nullptr;

    if (M % 32 == 0 && N % 32 == 0) {
        printf("\n**testing kernel_32x32_4x4_low_sparsity:\n");
        low_sparsity_kernel = &nmGEMM_small_matrices_low_sparsity;
        high_sparsity_kernel = &nmGEMM_small_matrices_high_sparsity;
        int Ns = 32;
        test_performance(low_sparsity_kernel, high_sparsity_kernel, Ns, M, N, K, pruning_M, sparsity, SPLIT_K, warm_up, iter);
    }

    if (M % 32 == 0 && N % 64 == 0) {
        printf("\n**testing kernel_32x64_8x4_low_sparsity:\n");
        low_sparsity_kernel = &nmGEMM_medium_matrices_low_sparsity;
        high_sparsity_kernel = &nmGEMM_medium_matrices_high_sparsity;
        int Ns = 64;
        test_performance(low_sparsity_kernel, high_sparsity_kernel, Ns, M, N, K, pruning_M, sparsity, SPLIT_K, warm_up, iter);
    }

    if (M % 64 == 0 && N % 64 == 0) {
        printf("\n**testing kernel_64x64_8x8_low_sparsity:\n");
        low_sparsity_kernel = &nmGEMM_large_matrices_low_sparsity;
        high_sparsity_kernel = &nmGEMM_large_matrices_high_sparsity;
        int Ns = 64;
        test_performance(low_sparsity_kernel, high_sparsity_kernel, Ns, M, N, K, pruning_M, sparsity, SPLIT_K, warm_up, iter);
    }

    if (M % 64 == 0 && N % 128 == 0) {
        printf("\n**testing kernel_64x128_8x8_low_sparsity:\n");
        low_sparsity_kernel = &nmGEMM_huge_matrices_low_sparsity;
        high_sparsity_kernel = &nmGEMM_huge_matrices_high_sparsity;
        int Ns = 128;
        test_performance(low_sparsity_kernel, high_sparsity_kernel, Ns, M, N, K, pruning_M, sparsity, SPLIT_K, warm_up, iter);
    }

    return 0;
}
