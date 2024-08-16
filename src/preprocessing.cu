#include "NM-SpMM.h"

void transIndex(int* DT, int W, int N, int Q, int& pruning_N, int& pruning_M)
{
    for (int j = 0; j < N; j += VEC_LEN) {
        int a = (int)(j / VEC_LEN);
        for (int k = pruning_N; k < W; k += pruning_N * 2) {
            for (int u = 0; u < pruning_N; ++u) {
                DT[(k + u) * Q + a] += pruning_M;
            }
        }
    }
    pruning_M *= 2;
    pruning_N *= 2;
}

void PreProcessing_high_sparsity(int* column_info, int pruning_M, int pruning_N, int* DT, int W, int Q, int N, int Ns)
{
    // re indexing
    int Qs = Ns / VEC_LEN;
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

void PreProcessing_low_sparsity(int* DT, int W, int Q, int Ns)
{
    int Qs = Ns / VEC_LEN;
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
