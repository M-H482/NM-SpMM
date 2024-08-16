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
        for (int k = 0; k < W; k += pruning_N) {

            std::random_shuffle(tmp_index, tmp_index + pruning_M);
            std::sort(tmp_index, tmp_index + pruning_N);

            for (int u = 0; u < pruning_N; ++u) {
                for (int iner_j = 0; iner_j < alignN; iner_j++) {
                    B_idx[(k + u) + (j + iner_j) * W] = tmp_index[u] + k / (1.0f - sparsity);
                    B_idxT[(k + u) * N + (j + iner_j)] = B_idx[(k + u) + (j + iner_j) * W];
                }
            }
        }
    }
    free(tmp_index);
}

__global__ void nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x128_4x4(float* g_vec, float* g_mat_data, int* g_mat_index, float* g_data, const int M, const int N, const int K, const float sparsity)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_K = 128;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;
    extern __shared__ float shared_mem[];
    const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));

    int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
    int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

    const int A_THREADS_PER_ROW = BLOCK_SIZE_M / 4;
    const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

    const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
    const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

    float* A_shared = shared_mem;
    float* B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE;

    float A_reg[THREAD_SIZE_M];
    float B_reg[THREAD_SIZE_N];
    float C_reg[THREAD_SIZE_N][THREAD_SIZE_M] = { 0 };

    int tid = threadIdx.x;

    int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
    int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

    int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

    for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE) {
        float* A_global_ptr = g_vec + M_BLOCK_START;
        float* B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
        int* B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES) {
            int idx = *(B_index_global_ptr + (i + A_BLOCK_ROW_START) * N);
            *(float4*)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_M + A_BLOCK_COL_START) = *(float4*)(A_global_ptr + idx * M + A_BLOCK_COL_START);
        }

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES) {
            *(float4*)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) = *(float4*)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += 1) {
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_M; k += 1) {
                A_reg[k] = A_shared[i * BLOCK_SIZE_M + t_M * THREAD_SIZE_M + k];
            }
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k += 1) {
                B_reg[k] = B_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
            }
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k += 1) {
#pragma unroll
                for (int j = 0; j < THREAD_SIZE_M; j += 1) {
                    C_reg[k][j] += B_reg[k] * A_reg[j];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < THREAD_SIZE_N; i += 1) {
#pragma unroll
        for (int j = 0; j < THREAD_SIZE_M; j += 1) {
            g_data[(BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + i) * M + (BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + j)] = C_reg[i][j];
        }
    }
}

__global__ void nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x256_4x4(float* g_vec, float* g_mat_data, int* g_mat_index, float* g_data, const int M, const int N, const int K, const float sparsity)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_K = 256;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;
    extern __shared__ float shared_mem[];
    const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));

    int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
    int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

    const int A_THREADS_PER_ROW = BLOCK_SIZE_M / 4;
    const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

    const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
    const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

    float* A_shared = shared_mem;
    float* B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE;

    float A_reg[THREAD_SIZE_M];
    float B_reg[THREAD_SIZE_N];
    float C_reg[THREAD_SIZE_N][THREAD_SIZE_M] = { 0 };

    int tid = threadIdx.x;

    int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
    int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

    int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

    for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE) {
        float* A_global_ptr = g_vec + M_BLOCK_START;
        float* B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
        int* B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES) {
            int idx = *(B_index_global_ptr + (i + A_BLOCK_ROW_START) * N);
            *(float4*)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_M + A_BLOCK_COL_START) = *(float4*)(A_global_ptr + idx * M + A_BLOCK_COL_START);
        }

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES) {
            *(float4*)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) = *(float4*)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += 1) {
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_M; k += 1) {
                A_reg[k] = A_shared[i * BLOCK_SIZE_M + t_M * THREAD_SIZE_M + k];
            }
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k += 1) {
                B_reg[k] = B_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
            }
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k += 1) {
#pragma unroll
                for (int j = 0; j < THREAD_SIZE_M; j += 1) {
                    C_reg[k][j] += B_reg[k] * A_reg[j];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < THREAD_SIZE_N; i += 1) {
#pragma unroll
        for (int j = 0; j < THREAD_SIZE_M; j += 1) {
            g_data[(BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + i) * M + (BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + j)] = C_reg[i][j];
        }
    }
}

void nmsparse(float* mat_a_dense, float* mat_b_sparse_val, int* mat_b_sparse_idx, float* output, int M, int K, int N, float sparsity)
{
    if (fabs(sparsity - 0.5f) < 1e-6) {
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_N = 32;
        const int BLOCK_SIZE_K = 128;
        const int THREAD_SIZE_M = 4;
        const int THREAD_SIZE_N = 4;
        const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
        dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
        dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
        int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
        nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x128_4x4<<<dimGrid, dimBlock, shared_mem_size>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, M, N, K, sparsity);
    } else {
        const int BLOCK_SIZE_M = 32;
        const int BLOCK_SIZE_N = 32;
        const int BLOCK_SIZE_K = 256;
        const int THREAD_SIZE_M = 4;
        const int THREAD_SIZE_N = 4;
        const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
        dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
        dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
        int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
        nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x256_4x4<<<dimGrid, dimBlock, shared_mem_size>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, M, N, K, sparsity);
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
                C[i + j * M] += A[i + B_idx[k + j * W] * M] * B[k + j * W];
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

    printf("M = %d, N = %d, K = %d, pruning_M = %d, sparsity = %f\n", M, N, K, pruning_M, sparsity);
    // ***************** initialize  *******************
    const int A_nBytes = sizeof(float) * M * K;
    const int C_nBytes = sizeof(float) * M * N;

    const int B_nBytes = sizeof(float) * W * N;
    const int B_idx_nBytes = sizeof(int) * W * N;

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
    // nmsparse(dA, dB, dB_idx, dC, M, K, N, sparsity);
    // cudaDeviceSynchronize();
    // cudaMemcpy(deviceRes, dC, C_nBytes, cudaMemcpyDeviceToHost);

    // if (allclose(deviceRes, hostRef, M * N)) {
    //     printf("The result is right!\n");
    // } else {
    //     printf("The result is wrong !!!!!!!!!!\n");
    // }

    // ***************** profling.    *******************

    for (int i = 0; i < 3; i++) {
        nmsparse(dA, dB, dB_idx, dC, M, K, N, sparsity);
    }

    // ***************** calculate performance *******************
    float milliseconds = 0.0f, tflops = -1.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warm_up; i++) {
        nmsparse(dA, dB, dB_idx, dC, M, K, N, sparsity);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < iter; i++) {
        nmsparse(dA, dB, dB_idx, dC, M, K, N, sparsity);
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
