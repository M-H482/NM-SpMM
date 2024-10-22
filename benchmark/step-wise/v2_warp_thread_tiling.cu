#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <time.h>
using namespace std;

// printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));

#define alignN 32
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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
                B_idxT[(k + u) * Q + a] = tmp_index[u];
            }
        }
    }
    free(tmp_index);
}

template <
    const int Ms,
    const int Ns,
    const int Ks,
    const int Ws,
    const int Mt,
    const int Nt>
__global__ void nmGEMM(float* A, float* B, int* D, float* C, int M, int N, int K, int W)
{
    /*
     *    A, B, D, C: col-major, row-major, row-major, row-major
     */
    const int Q = N / alignN;
    const int Qs = (Ns + alignN - 1) / alignN;

    extern __shared__ char smem[];
    float At[Mt], Bt[Nt], Ct[Mt][Nt] = { 0.0f };

    float* As = (float*)smem;
    float* Bs = (float*)(smem + Ks * Ms * sizeof(float));
    int* Ds = (int*)(smem + (Ks * Ms + Ws * Ns) * sizeof(float));

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;

    const int bi = blockIdx.y * Ms;
    const int bj = blockIdx.x * Ns;

    const int ti = ((lane_id % 16) / 2) * 4;
    const int tj = warp_id * 32 + (lane_id / 16) * 8 + (lane_id % 2) * 4;

    const int THREADS_PER_BLOCK = (Ms / Mt) * (Ns / Nt);

    const int A_THREADS_PER_ROW = Ms / 4;
    const int B_THREADS_PER_ROW = Ns / 4;
    const int D_THREADS_PER_ROW = Qs;

    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
    const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;
    const int D_STRIDES = THREADS_PER_BLOCK / D_THREADS_PER_ROW;

    int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;
    int D_BLOCK_ROW_START = tid / D_THREADS_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;
    int D_BLOCK_COL_START = tid % D_THREADS_PER_ROW;

    for (int u = 0, v = 0; u < W; u += Ws, v += Ks) {
        float* A_ptr = A + bi + v * M;
        float* B_ptr = B + bj + u * N;
        int* D_ptr = D + bj / alignN + u * Q; // 注意Ns不能整除alignN的情况
#pragma unroll
        for (int i = 0; i < Ks; i += A_STRIDES) {
            FETCH_FLOAT4(As[(i + A_BLOCK_ROW_START) * Ms + A_BLOCK_COL_START])
                = FETCH_FLOAT4(A_ptr[(i + A_BLOCK_ROW_START) * M + A_BLOCK_COL_START]);
        }
#pragma unroll
        for (int i = 0; i < Ws; i += B_STRIDES) {
            FETCH_FLOAT4(Bs[(i + B_BLOCK_ROW_START) * Ns + B_BLOCK_COL_START])
                = FETCH_FLOAT4(B_ptr[(i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START]);
        }
        for (int i = 0; i + D_BLOCK_ROW_START < Ws; i += D_STRIDES) {
            Ds[(i + D_BLOCK_ROW_START) * Qs + D_BLOCK_COL_START] = D_ptr[(i + D_BLOCK_ROW_START) * Q + D_BLOCK_COL_START];
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < Ws; p++) {
            FETCH_FLOAT4(At[0]) = FETCH_FLOAT4(As[Ds[p * Qs + tj / alignN] * Ms + ti]);
            FETCH_FLOAT4(At[4]) = FETCH_FLOAT4(As[Ds[p * Qs + tj / alignN] * Ms + ti + 32]);
            FETCH_FLOAT4(Bt[0]) = FETCH_FLOAT4(Bs[p * Ns + tj]);
            FETCH_FLOAT4(Bt[4]) = FETCH_FLOAT4(Bs[p * Ns + tj + 16]);
#pragma unroll
            for (int i = 0; i < Mt; i++) {
#pragma unroll
                for (int j = 0; j < Nt; j++) {
                    Ct[i][j] += At[i] * Bt[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(C[(bi + ti + i) * N + (bj + tj + 0)]) = FETCH_FLOAT4(Ct[i][0]);
        FETCH_FLOAT4(C[(bi + ti + i) * N + (bj + tj + 16)]) = FETCH_FLOAT4(Ct[i][4]);
        FETCH_FLOAT4(C[(bi + ti + i + 32) * N + (bj + tj + 0)]) = FETCH_FLOAT4(Ct[i + 4][0]);
        FETCH_FLOAT4(C[(bi + ti + i + 32) * N + (bj + tj + 16)]) = FETCH_FLOAT4(Ct[i + 4][4]);
    }
}

void nmspmm(float* A, float* B, int* D, float* C, int M, int N, int K, int W, float sparsity)
{
    const int Ms = 64;
    const int Ns = 128;
    const int Ks = 32;
    const int Mt = 8;
    const int Nt = 8;

    dim3 dimBlock(Ns / Nt, Ms / Mt);
    dim3 dimGrid(N / Ns, M / Ms);

    if (fabs(sparsity - 0.5f) < 1e-6) {
        const int Ws = 16;
        size_t smem_nbytes = (Ks * Ms + Ws * Ns) * sizeof(float)
            + (Ws * Ns / alignN) * sizeof(int);
        nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    } else if (fabs(sparsity - 0.625f) < 1e-6) {
        const int Ws = 12;
        size_t smem_nbytes = (Ks * Ms + Ws * Ns) * sizeof(float)
            + (Ws * Ns / alignN) * sizeof(int);
        nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    } else if (fabs(sparsity - 0.75f) < 1e-6) {
        const int Ws = 8;
        size_t smem_nbytes = (Ks * Ms + Ws * Ns) * sizeof(float)
            + (Ws * Ns / alignN) * sizeof(int);
        nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    } else if (fabs(sparsity - 0.875f) < 1e-6) {
        const int Ws = 4;
        size_t smem_nbytes = (Ks * Ms + Ws * Ns) * sizeof(float)
            + (Ws * Ns / alignN) * sizeof(int);
        nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    } else if (fabs(sparsity - 0.0f) < 1e-6) {
        const int Ws = 32;
        size_t smem_nbytes = (Ks * Ms + Ws * Ns) * sizeof(float)
            + (Ws * Ns / alignN) * sizeof(int);
        nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    }
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
    int warm_up = 100, iter = 100;

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
