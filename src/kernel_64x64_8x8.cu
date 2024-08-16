#include "NM-SpMM.h"
#include "ptx.h"
#include <cmath>

template <
    const int Ms,
    const int Ns,
    const int Ks,
    const int Ws,
    const int Mt,
    const int Nt>
__global__ void kernel_32x32_4x4_low_sparsity(float* A, float* B, int* D, float* C, int M, int N, int K, int W)
{
    /*
     *    A, B, D, C: col-major, row-major, row-major, row-major
     */
    int SPLIT_K = gridDim.z;
    int bz = blockIdx.z;
    const int Qs = (Ns + VEC_LEN - 1) / VEC_LEN;

    int K_LEN, SPLIT_K_OFFSET_A, SPLIT_K_OFFSET_B, SPLIT_K_OFFSET_D;
    int iter_num = W / Ws;
    if (iter_num % SPLIT_K == 0) {
        K_LEN = W / SPLIT_K;
        SPLIT_K_OFFSET_A = (iter_num / SPLIT_K) * Ks * M * bz;
        SPLIT_K_OFFSET_B = K_LEN * N * bz;
        SPLIT_K_OFFSET_D = K_LEN * Qs * bz;
    } else {
        int p = iter_num / SPLIT_K;
        int q = iter_num % SPLIT_K;
        int offset = (bz < q) ? (bz * p + bz) : (bz * p + q);

        K_LEN = ((bz < q) ? (p + 1) : p) * Ws;
        SPLIT_K_OFFSET_A = offset * Ks * M;
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
    int* D_ptr = D + bj / VEC_LEN * W + SPLIT_K_OFFSET_D;

    int idx[Ws];
    const int load_D_num = Ws * Qs;

#pragma unroll
    for (int i = 0; i < Ks; i += A_STRIDES) {
        FETCH_FLOAT4(As_write_ptr[(i + A_BLOCK_ROW_START) * Ms + A_BLOCK_COL_START])
            = FETCH_FLOAT4(A_ptr[(i + A_BLOCK_ROW_START) * M + A_BLOCK_COL_START]);
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
        idx[p] = Ds_write_ptr[p * Qs + tj / VEC_LEN];
    }

    FETCH_FLOAT4(Bt[0][0]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj]);
    FETCH_FLOAT4(Bt[0][4]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj + 16]);
    FETCH_FLOAT4(At[0][0]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti]);
    FETCH_FLOAT4(At[0][4]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti + 32]);

    for (int u = Ws, v = Ks; u < K_LEN; u += Ws, v += Ks) {

        A_ptr = A + bi + SPLIT_K_OFFSET_A + v * M;
        B_ptr = B + bj + SPLIT_K_OFFSET_B + u * N;
        D_ptr = D + bj / VEC_LEN * W + SPLIT_K_OFFSET_D + u * Qs;

        {
            float* t;
            t = As_read_ptr, As_read_ptr = As_write_ptr, As_write_ptr = t;
            t = Bs_read_ptr, Bs_read_ptr = Bs_write_ptr, Bs_write_ptr = t;
        }
        {
            int* t;
            t = Ds_read_ptr, Ds_read_ptr = Ds_write_ptr, Ds_write_ptr = t;
        }

        uint32_t addr = __cvta_generic_to_shared(&Ds_write_ptr[tid]);
        CP_ASYNC_CA_Guard(addr, D_ptr + tid, 4, tid < load_D_num);

#pragma unroll
        for (int i = 0; i < Ks; i += A_STRIDES) {
            uint32_t addr = __cvta_generic_to_shared(&As_write_ptr[(i + A_BLOCK_ROW_START) * Ms + A_BLOCK_COL_START]);
            CP_ASYNC_CG(addr, &A_ptr[(i + A_BLOCK_ROW_START) * M + A_BLOCK_COL_START], 16);
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
            idx[p] = Ds_write_ptr[p * Qs + tj / VEC_LEN];
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

void nmGEMM_large_matrices_low_sparsity(float* A, float* B, int* D, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K)
{
    const int Ms = 64;
    const int Ns = 64;
    const int Mt = 8;
    const int Nt = 8;

    dim3 dimBlock(Ns / Nt, Ms / Mt);
    dim3 dimGrid(N / Ns, M / Ms, SPLIT_K);

    if (fabs(sparsity - 0.75f) < 1e-6 || fabs(sparsity - 0.875f) < 1e-6) {
        printf("Not support! Please use low sparsity version for sparsity %.3f\n", sparsity);
    } else if (fabs(sparsity - 0.5f) < 1e-6) {
        const int Ks = 32;
        const int Ws = 16;
        size_t smem_nbytes = 2 * (Ks * Ms + Ws * Ns) * sizeof(float)
            + 2 * (Ws * Ns / VEC_LEN + Ks) * sizeof(int);
        kernel_32x32_4x4_low_sparsity<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    } else if (fabs(sparsity - 0.625f) < 1e-6) {
        const int Ks = 32;
        const int Ws = 12;
        size_t smem_nbytes = 2 * (Ks * Ms + Ws * Ns) * sizeof(float)
            + 2 * (Ws * Ns / VEC_LEN + Ks) * sizeof(int);
        kernel_32x32_4x4_low_sparsity<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    }
}

template <
    const int Ms,
    const int Ns,
    const int Ks,
    const int Ws,
    const int Mt,
    const int Nt>
__global__ void kernel_32x32_4x4_high_sparsity(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W)
{
    /*
     *    A, B, D, C: col-major, row-major, row-major, row-major
     */
    int SPLIT_K = gridDim.z;
    int bz = blockIdx.z;
    const int Qs = (Ns + VEC_LEN - 1) / VEC_LEN;

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
    int* D_ptr = D + bj / VEC_LEN * W + SPLIT_K_OFFSET_D;
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
        idx[p] = Ds_write_ptr[p * Qs + tj / VEC_LEN];
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
        D_ptr = D + bj / VEC_LEN * W + SPLIT_K_OFFSET_D + u * Qs;
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
            idx[p] = Ds_write_ptr[p * Qs + tj / VEC_LEN];
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

void nmGEMM_large_matrices_high_sparsity(float* A, float* B, int* D, int* column_info, float* C, int M, int N, int K, int W, float sparsity, int SPLIT_K)
{
    const int Ms = 64;
    const int Ns = 64;
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
            + 2 * (Ws * Ns / VEC_LEN + Ks) * sizeof(int);
        cudaFuncSetAttribute(kernel_32x32_4x4_high_sparsity<Ms, Ns, Ks, Ws, Mt, Nt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_nbytes);
        kernel_32x32_4x4_high_sparsity<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, column_info, C, M, N, K, W);
    } else if (fabs(sparsity - 0.875f) < 1e-6) {
        const int Ks = 64;
        const int Ws = 8;
        size_t smem_nbytes = 2 * (Ks * Ms + Ws * Ns) * sizeof(float)
            + 2 * (Ws * Ns / VEC_LEN + Ks) * sizeof(int);
        cudaFuncSetAttribute(kernel_32x32_4x4_high_sparsity<Ms, Ns, Ks, Ws, Mt, Nt>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_nbytes);
        kernel_32x32_4x4_high_sparsity<Ms, Ns, Ks, Ws, Mt, Nt>
            <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, column_info, C, M, N, K, W);
    }
}