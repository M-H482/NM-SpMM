#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
using namespace std;
// #include "include/half.hpp"

// using half_float::half;

#define CHECK_CUBLAS(Expr)                                         \
    {                                                              \
        int err = (Expr);                                          \
        if (err != 0) {                                            \
            printf("cuBLAS error %d at line %d\n", err, __LINE__); \
        }                                                          \
    }

void gemm(cublasHandle_t handle,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* beta,
    cudaDataType_t input_type,
    const void* A,
    const void* B,
    cudaDataType_t output_type,
    void* C,
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type,
#else
    cudaDataType_t compute_type,
#endif
    int algo)
{
    cublasStatus_t res = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        alpha, B, input_type, n, A, input_type, k,
        beta, C, output_type, n, compute_type, static_cast<cublasGemmAlgo_t>(algo));
    CHECK_CUBLAS(res);
}

int main(int argc, char* argv[])
{
    int test_num = 1;
    int warmup = 100, iter = 100;
    int m = 4096, n = 4096, k = 4096;

    if (argc == 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    } else if (argc == 5) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
        warmup = std::atoi(argv[4]);
        iter = std::atoi(argv[4]);
    }

    // const int test_num = 64;
    // int M_list[test_num];
    // int N_list[test_num];
    // int K_list[test_num];

    // for(int i = 0; i < test_num; ++i){
    //     M_list[i] = (i + 1) * 256;
    //     N_list[i] = (i + 1) * 256;
    //     K_list[i] = (i + 1) * 256;
    // }
    for (int i = 0; i < test_num; ++i) {
        // int m = M_list[i];
        // int n = N_list[i];
        // int k = K_list[i];
        float alpha = 1.0f;
        float beta = 0.0f;

        cudaDataType_t input_type = CUDA_R_32F;
        cudaDataType_t output_type = CUDA_R_32F;
#if __CUDACC_VER_MAJOR__ >= 11
        cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
        cudaDataType_t compute_type = CUDA_R_32F;
#endif
        // cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
        // cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        // cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO2_TENSOR_OP;
        // vector<cublasGemmAlgo_t> algos[17] = {
        //	CUBLAS_GEMM_ALGO0_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO1_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO2_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO3_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO4_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO5_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO6_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO7_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO8_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO9_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO10_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO11_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO12_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO13_TENSOR_OP,
        //     CUBLAS_GEMM_ALGO15_TENSOR_OP,
        //     CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        // };
        double gopss = 0.f;
        double latency = 1000.0;
        int start_algo_t_op = CUBLAS_GEMM_DEFAULT;
        int end_algo_t_op = CUBLAS_GEMM_DEFAULT;
        // int end_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo) {
            // cublasGemmAlgo_t algo = algos[i];

            // cudaSetDevice(0);
            void *A, *B, *C;
            cudaMalloc(&A, m * k * sizeof(float));
            cudaMalloc(&B, k * n * sizeof(float));
            cudaMalloc(&C, m * n * sizeof(float));

            cublasHandle_t handle;
            cublasCreate(&handle);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // warmup
            for (int i = 0; i < warmup; ++i) {
                gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
                    output_type, C, compute_type, algo);
            }
            cudaEventRecord(start);
            for (int i = 0; i < iter; ++i) {
                gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
                    output_type, C, compute_type, algo);
            }
            cudaEventRecord(stop);

            float time_ms = 0.f;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_ms, start, stop);

            long ops = (long)m * n * k * 2;
            double l = ((double)time_ms / iter / 1e3);
            double gops = ((double)ops / 1e12) / l;
            gopss = gops > gopss ? gops : gopss;
            latency = l < latency ? l : latency;
            cudaFree(A);
            cudaFree(B);
            cudaFree(C);
            // printf("CBLAS - M : %d, N : %d, K : %d, %f ms, %f Tflops\n", m, n, k, (time_ms/iter), gopss);
        }
        printf("latency: %f ms, TFLOPS: %f\n", latency, gopss);
    }
}
