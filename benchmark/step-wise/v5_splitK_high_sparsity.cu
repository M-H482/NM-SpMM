#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <time.h>
using namespace std;

// printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));

#define alignN 32
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CA_Guard(dst, src, Bytes, guard)                                \
    asm volatile(                                                                \
        "{.reg .pred p;\n"                                                       \
        " setp.ne.b32 p, %3, 0;\n"                                               \
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], %2; }\n" ::"r"(dst), \
        "l"(src), "n"(Bytes), "r"((int)(guard)))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG_Guard(dst, src, Bytes, guard)                                \
    asm volatile(                                                                \
        "{.reg .pred p;\n"                                                       \
        " setp.ne.b32 p, %3, 0;\n"                                               \
        " @p cp.async.cg.shared.global.L2::128B [%0], [%1], %2; }\n" ::"r"(dst), \
        "l"(src), "n"(Bytes), "r"((int)(guard)))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

__device__ inline void atomicAddFloat4(float* address, float* val)
{
    atomicAdd(address + 0, *(val + 0));
    atomicAdd(address + 1, *(val + 1));
    atomicAdd(address + 2, *(val + 2));
    atomicAdd(address + 3, *(val + 3));
}

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

    std::mt19937 gen(std::random_device {}());
    for (int j = 0; j < N; j += alignN) {
        int a = (int)(j / alignN);
        for (int k = 0; k < W; k += pruning_N) {

            std::shuffle(tmp_index, tmp_index + pruning_M, gen);
            // std::random_shuffle(tmp_index, tmp_index + pruning_M);
            std::sort(tmp_index, tmp_index + pruning_N);

            for (int u = 0; u < pruning_N; ++u) {
                // for(int iner_j = 0; iner_j < alignN; iner_j++){
                //     D[(k + u) + (j + iner_j) * W] = tmp_index[u] + k / (1.0f - sparsity);
                //     DT[(k + u) * N + (j + iner_j)] = D[(k + u) + (j + iner_j) * W];
                // }

                D[(k + u) + a * W] = tmp_index[u] + k / (1.0f - sparsity);
                // // D[(k + u) + a * W] = tmp_index[u];
                // DT[(k + u) * Q + a] = D[(k + u) + a * W];
                DT[(k + u) * Q + a] = tmp_index[u];
            }
        }
    }
    free(tmp_index);
}

void transIndex(int* DT, int W, int N, int Q, int& pruning_N, int& pruning_M)
{
    for (int j = 0; j < N; j += alignN) {
        int a = (int)(j / alignN);
        for (int k = pruning_N; k < W; k += pruning_N * 2) {
            for (int u = 0; u < pruning_N; ++u) {
                DT[(k + u) * Q + a] += pruning_M;
            }
        }
    }
    pruning_M *= 2;
    pruning_N *= 2;
}

void PreProcessing(int* column_info, int pruning_M, int pruning_N, int* DT, int W, int Q, int N, int Ns)
{
    // re indexing
    int Qs = Ns / alignN;
    int* column_info_ptr = column_info;

    // indices reordering & get colinfo
    for (int j = 0; j < Q; j += Qs) {
        for (int i = 0; i < W; i += pruning_N) {
            for (int x = 0; x < pruning_M; x++)
                column_info_ptr[x] = -1;

            int bucket[pruning_M];
            for (int x = 0; x < pruning_M; x++)
                bucket[x] = 0;

            for (int x = 0; x < pruning_N; x++) {
                for (int y = 0; y < Qs; y++) {
                    int v = DT[(i + x) * Q + (j + y)];
                    bucket[v] += 1;
                }
            }

            int map[pruning_M];
            for (int x = 0, y = 0; x < pruning_M; x++) {
                int v = bucket[x];
                if (v > 0) {
                    column_info_ptr[y] = x;
                    map[x] = y;
                    y++;
                }
            }

            for (int x = 0; x < pruning_N; x++) {
                for (int y = 0; y < Qs; y++) {
                    int v = DT[(i + x) * Q + (j + y)];
                    DT[(i + x) * Q + (j + y)] = map[v];
                }
            }

            column_info_ptr = column_info_ptr + pruning_M;
        }
    }

    // layout transform
    int* buffer = (int*)malloc(sizeof(int) * W * Q);
    for (int j = 0; j < Q; j += Qs) {
        int* p = buffer + j * W;
        for (int row = 0; row < W; row++) {
            for (int col = 0; col < Qs; col++) {
                *p = DT[row * Q + j + col];
                p += 1;
            }
        }
    }
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < Q; j++) {
            DT[i * Q + j] = buffer[i * Q + j];
        }
    }
    free(buffer);
}

template <
    const int Ms,
    const int Ns,
    const int Ks,
    const int Ws,
    const int Mt,
    const int Nt>
__global__ void nmGEMM(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W)
{
    /*
     *    A, B, D, C: col-major, row-major, row-major, row-major
     */
    int SPLIT_K = gridDim.z;
    int bz = blockIdx.z;
    const int Qs = (Ns + alignN - 1) / alignN;

    int K_LEN, SPLIT_K_OFFSET_A, SPLIT_K_OFFSET_B, SPLIT_K_OFFSET_D, SPLIT_K_OFFSET_I;
    int iter_num = W / Ws;
    if (iter_num % SPLIT_K == 0) {
        K_LEN = W / SPLIT_K;
        SPLIT_K_OFFSET_A = (iter_num / SPLIT_K) * Ks * M * bz;
        SPLIT_K_OFFSET_I = (iter_num / SPLIT_K) * Ks * 1 * bz;
        SPLIT_K_OFFSET_B = K_LEN * N * bz;
        SPLIT_K_OFFSET_D = K_LEN * Qs * bz;
    } else {
        // 20个任务均匀分为8份：3 3 3 3 2 2 2 2
        int p = iter_num / SPLIT_K;
        int q = iter_num % SPLIT_K;
        int offset = (bz < q) ? (bz * p + bz) : (bz * p + q);

        K_LEN = ((bz < q) ? (p + 1) : p) * Ws;
        SPLIT_K_OFFSET_A = offset * Ks * M;
        SPLIT_K_OFFSET_I = offset * Ks * 1;
        SPLIT_K_OFFSET_B = offset * Ws * N;
        SPLIT_K_OFFSET_D = offset * Ws * Qs;
    }

    extern __shared__ char smem[];
    float At[2][Mt], Bt[2][Nt], Ct[Mt][Nt] = { 0.0f };

    float* As_write_ptr = (float*)smem; // [Ks][Ms]
    float* As_read_ptr = As_write_ptr + Ks * Ms;

    float* Bs_write_ptr = (float*)(smem + 2 * Ks * Ms * sizeof(float));
    float* Bs_read_ptr = Bs_write_ptr + Ws * Ns; // [Ws][Ns]

    int* Ds_write_ptr = (int*)(smem + 2 * (Ks * Ms + Ws * Ns) * sizeof(float));
    int* Ds_read_ptr = Ds_write_ptr + Ws * Qs; // [Ws][Qs]

    int* col_info_write_ptr = (int*)(smem + 2 * (Ks * Ms + Ws * Ns) * sizeof(float) + 2 * Ws * Qs * sizeof(int));
    int* col_info_read_ptr = col_info_write_ptr + Ks; // [Ks]

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

    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
    const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

    int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

    float* A_ptr = A + bi + SPLIT_K_OFFSET_A;
    float* B_ptr = B + bj + SPLIT_K_OFFSET_B;
    int* D_ptr = D + bj / alignN * W + SPLIT_K_OFFSET_D;
    int* column_info_ptr = column_info + blockIdx.x * (W / Ws) * Ks + SPLIT_K_OFFSET_I;

    int idx[Ws], col[Ks];
    const int load_D_num = Ws * Qs;

    if (tid < Ks) {
        col_info_write_ptr[tid] = column_info_ptr[tid + Ks];
        col_info_read_ptr[tid] = column_info_ptr[tid];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Ks; i += A_STRIDES) {
        if (col_info_read_ptr[i + A_BLOCK_ROW_START] != -1) {
            FETCH_FLOAT4(As_write_ptr[(i + A_BLOCK_ROW_START) * Ms + A_BLOCK_COL_START])
                = FETCH_FLOAT4(A_ptr[col_info_read_ptr[i + A_BLOCK_ROW_START] * M + A_BLOCK_COL_START]);
        }
    }
#pragma unroll
    for (int i = 0; i < Ws; i += B_STRIDES) {
        FETCH_FLOAT4(Bs_write_ptr[(i + B_BLOCK_ROW_START) * Ns + B_BLOCK_COL_START])
            = FETCH_FLOAT4(B_ptr[(i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START]);
    }
    if (tid < load_D_num) {
        Ds_write_ptr[tid] = D_ptr[tid];
    }

    __syncthreads();
#pragma unroll
    for (int p = 0; p < Ws; p++) {
        idx[p] = Ds_write_ptr[p * Qs + tj / alignN];
    }
#pragma unroll
    for (int i = 0; i < Ks; i += A_STRIDES) {
        col[i] = col_info_write_ptr[i + A_BLOCK_ROW_START];
    }

    FETCH_FLOAT4(Bt[0][0]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj]);
    FETCH_FLOAT4(Bt[0][4]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj + 16]);
    FETCH_FLOAT4(At[0][0]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti]);
    FETCH_FLOAT4(At[0][4]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti + 32]);

    for (int u = Ws, v = Ks; u < K_LEN; u += Ws, v += Ks) {

        A_ptr = A + bi + SPLIT_K_OFFSET_A + v * M;
        B_ptr = B + bj + SPLIT_K_OFFSET_B + u * N;
        D_ptr = D + bj / alignN * W + SPLIT_K_OFFSET_D + u * Qs;
        column_info_ptr = column_info + blockIdx.x * (W / Ws) * Ks + SPLIT_K_OFFSET_I + v + Ks;

        {
            float* t;
            t = As_read_ptr, As_read_ptr = As_write_ptr, As_write_ptr = t;
            t = Bs_read_ptr, Bs_read_ptr = Bs_write_ptr, Bs_write_ptr = t;
        }
        {
            int* t;
            t = Ds_read_ptr, Ds_read_ptr = Ds_write_ptr, Ds_write_ptr = t;
            t = col_info_read_ptr, col_info_read_ptr = col_info_write_ptr, col_info_write_ptr = t;
        }

        {
            uint32_t addr = __cvta_generic_to_shared(&col_info_write_ptr[tid]);
            CP_ASYNC_CA_Guard(addr, column_info_ptr + tid, 4, tid < Ks);
        }

        {
            uint32_t addr = __cvta_generic_to_shared(&Ds_write_ptr[tid]);
            CP_ASYNC_CA_Guard(addr, D_ptr + tid, 4, tid < load_D_num);
        }

#pragma unroll
        for (int i = 0; i < Ks; i += A_STRIDES) {
            uint32_t addr = __cvta_generic_to_shared(&As_write_ptr[(i + A_BLOCK_ROW_START) * Ms + A_BLOCK_COL_START]);
            CP_ASYNC_CG_Guard(addr, &A_ptr[col[i] * M + A_BLOCK_COL_START], 16, col[i] != -1);
        }
#pragma unroll
        for (int i = 0; i < Ws; i += B_STRIDES) {
            uint32_t addr = __cvta_generic_to_shared(&Bs_write_ptr[(i + B_BLOCK_ROW_START) * Ns + B_BLOCK_COL_START]);
            CP_ASYNC_CG(addr, &B_ptr[(i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START], 16);
        }

        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int p = 0; p < Ws - 1; p += 1) {
            FETCH_FLOAT4(Bt[(p + 1) % 2][0]) = FETCH_FLOAT4(Bs_read_ptr[(p + 1) * Ns + tj]);
            FETCH_FLOAT4(Bt[(p + 1) % 2][4]) = FETCH_FLOAT4(Bs_read_ptr[(p + 1) * Ns + tj + 16]);
            FETCH_FLOAT4(At[(p + 1) % 2][0]) = FETCH_FLOAT4(As_read_ptr[idx[p + 1] * Ms + ti]);
            FETCH_FLOAT4(At[(p + 1) % 2][4]) = FETCH_FLOAT4(As_read_ptr[idx[p + 1] * Ms + ti + 32]);
#pragma unroll
            for (int i = 0; i < Mt; i++) {
                if (i % 2) {
#pragma unroll
                    for (int j = Nt - 1; j >= 0; j--) {
                        Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < Nt; j++) {
                        Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                    }
                }
            }
        }
        CP_ASYNC_WAIT_ALL();
        __syncthreads();

#pragma unroll
        for (int p = 0; p < Ws; p++) {
            idx[p] = Ds_write_ptr[p * Qs + tj / alignN];
        }
#pragma unroll
        for (int i = 0; i < Ks; i += A_STRIDES) {
            col[i] = col_info_write_ptr[i + A_BLOCK_ROW_START];
        }

        FETCH_FLOAT4(Bt[0][0]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj]);
        FETCH_FLOAT4(Bt[0][4]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj + 16]);
        FETCH_FLOAT4(At[0][0]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti]);
        FETCH_FLOAT4(At[0][4]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti + 32]);

#pragma unroll
        for (int i = 0; i < Mt; i++) {
            if (i % 2) {
#pragma unroll
                for (int j = Nt - 1; j >= 0; j--) {
                    Ct[i][j] += At[1][i] * Bt[1][j];
                }
            } else {
#pragma unroll
                for (int j = 0; j < Nt; j++) {
                    Ct[i][j] += At[1][i] * Bt[1][j];
                }
            }
        }
    }

    {
        float* t;
        t = As_read_ptr, As_read_ptr = As_write_ptr, As_write_ptr = t;
        t = Bs_read_ptr, Bs_read_ptr = Bs_write_ptr, Bs_write_ptr = t;
    }
    {
        int* t;
        t = Ds_read_ptr, Ds_read_ptr = Ds_write_ptr, Ds_write_ptr = t;
        t = col_info_read_ptr, col_info_read_ptr = col_info_write_ptr, col_info_write_ptr = t;
    }

#pragma unroll
    for (int p = 0; p < Ws - 1; p++) {
        FETCH_FLOAT4(Bt[(p + 1) % 2][0]) = FETCH_FLOAT4(Bs_read_ptr[(p + 1) * Ns + tj]);
        FETCH_FLOAT4(Bt[(p + 1) % 2][4]) = FETCH_FLOAT4(Bs_read_ptr[(p + 1) * Ns + tj + 16]);
        FETCH_FLOAT4(At[(p + 1) % 2][0]) = FETCH_FLOAT4(As_read_ptr[idx[p + 1] * Ms + ti]);
        FETCH_FLOAT4(At[(p + 1) % 2][4]) = FETCH_FLOAT4(As_read_ptr[idx[p + 1] * Ms + ti + 32]);
#pragma unroll
        for (int i = 0; i < Mt; i++) {
            if (i % 2) {
#pragma unroll
                for (int j = Nt - 1; j >= 0; j--) {
                    Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                }
            } else {
#pragma unroll
                for (int j = 0; j < Nt; j++) {
                    Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Mt; i++) {
        if (i % 2) {
#pragma unroll
            for (int j = Nt - 1; j >= 0; j--) {
                Ct[i][j] += At[1][i] * Bt[1][j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < Nt; j++) {
                Ct[i][j] += At[1][i] * Bt[1][j];
            }
        }
    }

    if (SPLIT_K > 1) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            atomicAddFloat4(&C[(bi + ti + i) * N + (bj + tj + 0)], &Ct[i][0]);
            atomicAddFloat4(&C[(bi + ti + i) * N + (bj + tj + 16)], &Ct[i][4]);
            atomicAddFloat4(&C[(bi + ti + i + 32) * N + (bj + tj + 0)], &Ct[i + 4][0]);
            atomicAddFloat4(&C[(bi + ti + i + 32) * N + (bj + tj + 16)], &Ct[i + 4][4]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            FETCH_FLOAT4(C[(bi + ti + i) * N + (bj + tj + 0)]) = FETCH_FLOAT4(Ct[i][0]);
            FETCH_FLOAT4(C[(bi + ti + i) * N + (bj + tj + 16)]) = FETCH_FLOAT4(Ct[i][4]);
            FETCH_FLOAT4(C[(bi + ti + i + 32) * N + (bj + tj + 0)]) = FETCH_FLOAT4(Ct[i + 4][0]);
            FETCH_FLOAT4(C[(bi + ti + i + 32) * N + (bj + tj + 16)]) = FETCH_FLOAT4(Ct[i + 4][4]);
        }
    }
}

void nmspmm(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K)
{
    const int Ms = 64;
    const int Ns = 128;
    const int Mt = 8;
    const int Nt = 8;

    dim3 dimBlock(Ns / Nt, Ms / Mt);
    dim3 dimGrid(N / Ns, M / Ms, SPLIT_K);

    if (fabs(sparsity - 0.5f) < 1e-6 || fabs(sparsity - 0.625f) < 1e-6) {
        printf("Not support! Please use low sparsity version for sparsity %.2f\n", sparsity);
    } else if (fabs(sparsity - 0.75f) < 1e-6) {
        const int Ks = 64;
        const int Ws = 16;
        size_t smem_nbytes = 2 * (Ks * Ms + Ws * Ns) * sizeof(float)
            + 2 * (Ws * Ns / alignN + Ks) * sizeof(int);
        cudaFuncSetAttribute(nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_nbytes);
        nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, column_info, C, M, N, K, W);
    } else if (fabs(sparsity - 0.875f) < 1e-6) {
        const int Ks = 64;
        const int Ws = 8;
        size_t smem_nbytes = 2 * (Ks * Ms + Ws * Ns) * sizeof(float)
            + 2 * (Ws * Ns / alignN + Ks) * sizeof(int);
        cudaFuncSetAttribute(nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_nbytes);
        nmGEMM<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, column_info, C, M, N, K, W);
    }
}

void matmul_on_cpu(float* A, float* B, int* D, float* C, int M, int N, int K, int W)
{
    int num_threads = omp_get_max_threads();
    printf("Using %d threads compute reference on CPU\n", num_threads);
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < W; k++) {
            for (int j = 0; j < N; j++) {
                int a = (int)(j / alignN);
                C[i * N + j] += A[i + D[k + a * W] * M] * B[k + j * W];
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
    int SPLIT_K = 1;
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
    } else if (argc == 7) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        pruning_M = atoi(argv[4]);
        sparsity = atof(argv[5]);
        SPLIT_K = atoi(argv[6]);
    } else if (argc == 9) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        pruning_M = atoi(argv[4]);
        sparsity = atof(argv[5]);
        SPLIT_K = atoi(argv[6]);
        warm_up = atoi(argv[7]);
        iter = atoi(argv[8]);
    }

    int W = (int)(K * (1.0f - sparsity));
    int Q = (int)(N / alignN);
    int pruning_N = (int)(pruning_M * (1.0f - sparsity));

    printf("M = %d, N = %d, K = %d, pruning_M = %d, sparsity = %f, SPLIT_K = %d\n", M, N, K, pruning_M, sparsity, SPLIT_K);
    // ***************** initialize  *******************
    const int A_nBytes = sizeof(float) * M * K;
    const int C_nBytes = sizeof(float) * M * N;

    const int B_nBytes = sizeof(float) * W * N;
    const int D_nBytes = sizeof(int) * W * Q;

    float* hA = (float*)malloc(A_nBytes);
    float* hB = (float*)malloc(B_nBytes);
    float* hB_T = (float*)malloc(B_nBytes);
    float* hC = (float*)malloc(C_nBytes);

    float* hostRef = (float*)malloc(C_nBytes);
    float* deviceRes = (float*)malloc(C_nBytes);

    int* hD = (int*)malloc(D_nBytes);
    int* hD_T = (int*)malloc(D_nBytes);

    init_data(hA, hB, hD, hB_T, hD_T, hC, M, N, K, pruning_M, sparsity);
    if (pruning_M == 32) {
        transIndex(hD_T, W, N, Q, pruning_N, pruning_M);
    }

    int Ns = 128;
    int column_nBytes = ((W / pruning_N) * (N / Ns) + 1) * (pruning_M) * sizeof(int);
    int* column_info = (int*)malloc(column_nBytes);

    PreProcessing(column_info, pruning_M, pruning_N, hD_T, W, Q, N, Ns);
    int cnt = 0;
    for (int i = 0; i < (W / pruning_N) * (N / Ns) * (pruning_M); i++) {
        if (column_info[i] == -1) {
            cnt++;
        }
    }
    printf("Reduce %f\n", 1.0f * cnt / ((W / pruning_N) * (N / Ns) * (pruning_M)));

    float *dA, *dB, *dC;
    int *dD, *dcolumn_info;
    cudaMalloc((void**)&dA, A_nBytes);
    cudaMalloc((void**)&dB, B_nBytes);
    cudaMalloc((void**)&dC, C_nBytes);
    cudaMalloc((void**)&dD, D_nBytes);
    cudaMalloc((void**)&dcolumn_info, column_nBytes);

    cudaMemcpy(dA, hA, A_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_T, B_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, C_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dD, hD_T, D_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dcolumn_info, column_info, column_nBytes, cudaMemcpyHostToDevice);

    // ***************** result check *******************

    // matmul_on_cpu(hA, hB, hD, hostRef, M, N, K, W);
    // // trans_inplace(hostRef, M, N);
    // nmspmm(dA, dB, dD, dcolumn_info, dC, M, N, K, W, sparsity, SPLIT_K);
    // cudaDeviceSynchronize();
    // cudaMemcpy(deviceRes, dC, C_nBytes, cudaMemcpyDeviceToHost);

    // if (allclose(deviceRes, hostRef, M * N)) {
    //     printf("The result is right!\n");
    // } else {
    //     printf("The result is wrong !!!!!!!!!!\n");
    //     return 0;
    // }

    // ***************** profling.    *******************

    for (int i = 0; i < 3; i++) {
        nmspmm(dA, dB, dD, dcolumn_info, dC, M, N, K, W, sparsity, SPLIT_K);
    }

    // ***************** calculate performance *******************
    float milliseconds = 0.0f, tflops = -1.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warm_up; i++) {
        nmspmm(dA, dB, dD, dcolumn_info, dC, M, N, K, W, sparsity, SPLIT_K);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < iter; i++) {
        nmspmm(dA, dB, dD, dcolumn_info, dC, M, N, K, W, sparsity, SPLIT_K);
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
