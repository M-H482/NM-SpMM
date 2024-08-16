#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <time.h>
using namespace std;

#define alignN 32

void init_data(float* A, float* B, int* B_idx, float* BT, int* B_idxT, float* C, int M, int N, int K, int pruning_M, float sparsity)
{
    /**
     *  A:      col-major
     *  B:      col-major
     *  B_idx:  col-major
     *  BT:     row-major
     *  B_idxT: row-major
     */

    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    // srand(1);
    const unsigned int W = K * (1.0f - sparsity);
    const unsigned int pruning_N = pruning_M * (1.0f - sparsity);
    const unsigned int Q = (int)(N / alignN);

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

    for (int j = 0; j < N; j += alignN) {
        int a = (int)(j / alignN);
        for (int k = 0; k < W; k += pruning_N) {

            std::random_shuffle(tmp_index, tmp_index + pruning_M);
            std::sort(tmp_index, tmp_index + pruning_N);

            for (int u = 0; u < pruning_N; ++u) {
                // for(int iner_j = 0; iner_j < alignN; iner_j++){
                //     B_idx[(k + u) + (j + iner_j) * W] = tmp_index[u] + k / (1.0f - sparsity);
                //     B_idxT[(k + u) * N + (j + iner_j)] = B_idx[(k + u) + (j + iner_j) * W];
                // }

                B_idx[(k + u) + a * W] = tmp_index[u] + k / (1.0f - sparsity);
                // // B_idx[(k + u) + a * W] = tmp_index[u];
                // B_idxT[(k + u) * Q + a] = B_idx[(k + u) + a * W];
                B_idxT[(k + u) * Q + a] = tmp_index[u] + k / (1.0f - sparsity);
            }
        }
    }
    free(tmp_index);
}

template <
    const int Ms, // height of block of C that each thread block calculate
    const int Ns // width of block of C that each thread block load into shared memory
    >
__global__ void nmGEMM(float* A, float* B, int* D, float* C, int M, int N, int K, int W)
{
    /*
     *    A, B, D, C: col-major, row-major, row-major, row-major
     */
    int tid = threadIdx.x;
    int tx = tid % Ns;
    int ty = tid / Ns;

    int i = blockIdx.y * Ms + ty;
    int j = blockIdx.x * Ns + tx;
    float sum = 0.0f;

    const int Q = N / alignN;
    for (int u = 0; u < W; u++) {
        int t = D[u * Q + j / alignN];
        sum += A[i + t * M] * B[u * N + j];
    }
    C[i * N + j] = sum;
}

void nmspmm(float* A, float* B, int* B_idx, float* C, int M, int N, int K, int W, float sparsity)
{
    const int Ms = 32;
    const int Ns = 32;

    dim3 dimBlock(Ms * Ns);
    dim3 dimGrid(N / Ns, M / Ms);

    nmGEMM<Ms, Ns>
        <<<dimGrid, dimBlock>>>(A, B, B_idx, C, M, N, K, W);
}

void matmul_on_cpu(float* A, float* B, int* B_idx, float* C, int M, int N, int K, int W)
{
    int num_threads = omp_get_max_threads();
    printf("Using %d threads compute reference on CPU\n", num_threads);
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < W; k++) {
            for (int j = 0; j < N; j++) {
                int a = (int)(j / alignN);
                C[i * N + j] += A[i + B_idx[k + a * W] * M] * B[k + j * W];
            }
        }
    }
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

void trans_inplace(float* a, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            swap(a[i * n + j], a[i + j * m]);
        }
    }
}

int main(int argc, char** argv)
{
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int pruning_M = 32;
    float sparsity = 0.5f;
    int warm_up = 10, iter = 10;

    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    } else if (argc == 6) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        pruning_M = atoi(argv[4]);
        sparsity = atof(argv[5]);
    } else if (argc == 8) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        pruning_M = atoi(argv[4]);
        sparsity = atof(argv[5]);
        warm_up = atoi(argv[6]);
        iter = atoi(argv[7]);
    }

    int W = (int)(K * (1.0f - sparsity));
    int Q = (int)(N / alignN);

    printf("M = %d, N = %d, K = %d, pruning_M = %d, sparsity = %f\n", M, N, K, pruning_M, sparsity);
    // ***************** initialize  *******************
    const int A_nBytes = sizeof(float) * M * K;
    const int C_nBytes = sizeof(float) * M * N;

    const int B_nBytes = sizeof(float) * W * N;
    const int B_idx_nBytes = sizeof(int) * W * Q;

    float* hA = (float*)malloc(A_nBytes);
    float* hB = (float*)malloc(B_nBytes);
    float* hB_T = (float*)malloc(B_nBytes);
    float* hC = (float*)malloc(C_nBytes);

    float* hostRef = (float*)malloc(C_nBytes);
    float* deviceRes = (float*)malloc(C_nBytes);

    int* hB_idx = (int*)malloc(B_idx_nBytes);
    int* hB_T_idx = (int*)malloc(B_idx_nBytes);

    init_data(hA, hB, hB_idx, hB_T, hB_T_idx, hC, M, N, K, pruning_M, sparsity);

    float *dA, *dB, *dC;
    int* dB_idx;
    cudaMalloc((void**)&dA, A_nBytes);
    cudaMalloc((void**)&dB, B_nBytes);
    cudaMalloc((void**)&dC, C_nBytes);
    cudaMalloc((void**)&dB_idx, B_idx_nBytes);

    cudaMemcpy(dA, hA, A_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_T, B_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, C_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_idx, hB_T_idx, B_idx_nBytes, cudaMemcpyHostToDevice);

    // ***************** result check *******************

    // matmul_on_cpu(hA, hB, hB_idx, hostRef, M, N, K, W);
    // // trans_inplace(hostRef, M, N);
    // nmspmm(dA, dB, dB_idx, dC, M, N, K, W, sparsity);
    // cudaDeviceSynchronize();
    // cudaMemcpy(deviceRes, dC, C_nBytes, cudaMemcpyDeviceToHost);

    // if (allclose(deviceRes, hostRef, M * N)) {
    //     printf("The result is right!\n");
    // } else {
    //     printf("The result is wrong !!!!!!!!!!\n");
    // }

    // ***************** profling.    *******************

    for (int i = 0; i < 3; i++) {
        nmspmm(dA, dB, dB_idx, dC, M, N, K, W, sparsity);
    }

    // ***************** calculate performance *******************
    float milliseconds = 0.0f, tflops = -1.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warm_up; i++) {
        nmspmm(dA, dB, dB_idx, dC, M, N, K, W, sparsity);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < iter; i++) {
        nmspmm(dA, dB, dB_idx, dC, M, N, K, W, sparsity);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds = milliseconds / iter;

    tflops = (2.0f * M * N * K / 1e12) / (milliseconds / 1e3);
    printf("Time elapsed: %f ms, %f TFLOPS\n", milliseconds, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
