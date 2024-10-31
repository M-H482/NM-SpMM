# NM-SpMM:Efficient N:M sparsity implementation for GPGPU

Deep learning has demonstrated effectiveness across a wide range of tasks.
However, the dense and over-parameterized nature of these models results in significant resource consumption during deployment. 
In response to this issue, weight pruning, particularly through N:M sparsity matrix multiplication, offers an efficient solution by transforming dense operations into semi-sparse ones.
N:M sparsity provides an option for balancing performance and model accuracy, but introduces more complex programming and optimization challenges.
To address these issues, we designed a systematic top-down performance analysis model for N:M sparsity.
Meanwhile, NM-SpMM is proposed as an efficient general N:M sparsity implementation. 
Based on our performance analysis, NM-SpMM employs a hierarchical blocking mechanism as a general optimization to enhance data locality, while memory access optimization and pipeline design are introduced as sparsity-aware optimizations, allowing it to achieve close-to-theoretical peak performance across different sparsity levels.
Experimental results show that NM-SpMM is 2.1x faster than nmSPARSE (the state-of-the-art for general N:M sparsity) and 1.4x to 6.3x faster than cuBLAS’s dense GEMM operations, closely approaching the theoretical maximum speedup resulting from the reduction in computation due to sparsity.
***
step-wise experiment, `-a` selects the SM architecture, where A100, 3090, and 4090 correspond to `sm_80`, `sm_86`, and `sm_89`, respectively. 
```shell
cd benchmark/step-wise && bash run_bench.sh -a sm_80
```

different kernels experiment
```shell
cd benchmark/different_kernel && bash run_nmspmm.sh
```

kernel performance experiment see [kernel performance README](./benchmark/kernel_performance/README.md).