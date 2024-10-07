The performance testing of NM-SpMM can be found in `kernel_performance_sparsity_50.0.pdf`, `kernel_performance_sparsity_62.5.pdf`, `kernel_performance_sparsity_75.0.pdf`, and `kernel_performance_sparsity_87.5.pdf`. We selected three baselines: cuBLAS, [nmSPARSE](https://github.com/LeiWang1999/nmsparse.git), and [Sputnik](https://github.com/zheng-ningxin/sputnik.git). cuBLAS represents dense computation, Sputnik uses an unstructured sparsity pattern, while both nmSPARSE and our NM-SpMM utilize the N:M sparsity pattern with M set to 32. The machines we evaluated include the A100 PCIe 80GB and the 3090 and 4090 GPUs.

Here is the current directory structure: the folder contains performance test results images, plotting scripts, and code. Additionally, there are two directories: `run_test`, which includes the code and compilation instructions for each baseline, and `result_data`, which is used to store the test result data.

Anyone can use the `build.sh` script in the corresponding directory to compile the executable files. Then, they can either choose `run_local_job.sh` to run the job locally on their machine or use `run_slurm_job.sh` to submit the job to a Slurm cluster (note that job parameters need to be modified as needed). This may be less friendly for users who are not on Slurm, and we sincerely apologize for that.

```plaintext
├── kernel_performance_sparsity_50.0.pdf
├── kernel_performance_sparsity_62.5.pdf
├── kernel_performance_sparsity_75.0.pdf
├── kernel_performance_sparsity_87.5.pdf
├── plot.py
├── plot.sh
├── README.md
├── result_data
│   ├── cublas_3090.txt
│   ├── cublas_4090.txt
│   ├── cublas_a100.txt
│   ├── nmsparse_3090_0.5.txt
│   ├── nmsparse_3090_0.625.txt
│   ├── nmsparse_3090_0.75.txt
│   ├── nmsparse_3090_0.875.txt
│   ├── nmsparse_4090_0.5.txt
│   ├── nmsparse_4090_0.625.txt
│   ├── nmsparse_4090_0.75.txt
│   ├── nmsparse_4090_0.875.txt
│   ├── nmsparse_a100_0.5.txt
│   ├── nmsparse_a100_0.625.txt
│   ├── nmsparse_a100_0.75.txt
│   ├── nmsparse_a100_0.875.txt
│   ├── nmspmm_3090_0.5.txt
│   ├── nmspmm_3090_0.625.txt
│   ├── nmspmm_3090_0.75.txt
│   ├── nmspmm_3090_0.875.txt
│   ├── nmspmm_4090_0.5.txt
│   ├── nmspmm_4090_0.625.txt
│   ├── nmspmm_4090_0.75.txt
│   ├── nmspmm_4090_0.875.txt
│   ├── nmspmm_a100_0.5.txt
│   ├── nmspmm_a100_0.625.txt
│   ├── nmspmm_a100_0.75.txt
│   ├── nmspmm_a100_0.875.txt
│   ├── sputnik_3090_0.5.txt
│   ├── sputnik_3090_0.625.txt
│   ├── sputnik_3090_0.75.txt
│   ├── sputnik_3090_0.875.txt
│   ├── sputnik_4090_0.5.txt
│   ├── sputnik_4090_0.625.txt
│   ├── sputnik_4090_0.75.txt
│   ├── sputnik_4090_0.875.txt
│   ├── sputnik_a100_0.5.txt
│   ├── sputnik_a100_0.625.txt
│   ├── sputnik_a100_0.75.txt
│   └── sputnik_a100_0.875.txt
└── run_test
    ├── cublas
    │   ├── build.sh
    │   ├── cublas_f32
    │   ├── cublas_f32.cu
    │   ├── run_cublas.sh
    │   ├── run_local_job.sh
    │   └── run_slurm_job.sh
    ├── nmsparse
    │   ├── build.sh
    │   ├── nmsparse
    │   ├── nmsparse.cu
    │   ├── run_local_job.sh
    │   ├── run_nmsparse.sh
    │   └── run_slurm_job.sh
    ├── nmspmm
    │   ├── Makefile
    │   ├── run_local_job.sh
    │   ├── run_nmspmm.sh
    │   ├── run_slurm_job.sh
    │   ├── test_nmspmm
    │   └── test_nmspmm.cu
    └── sputnik
        ├── build.sh
        ├── env.sh
        ├── run_local_job.sh
        ├── run_slurm_job.sh
        ├── run_sputnik.sh
        ├── test_sputnik
        └── test_sputnik.cu
```