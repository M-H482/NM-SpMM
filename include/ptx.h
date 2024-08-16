#ifndef PTX_H
#define PTX_H

#include <cuda.h>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

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

#endif // PTX_H
