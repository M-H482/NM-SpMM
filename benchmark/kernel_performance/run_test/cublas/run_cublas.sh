#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1   
#SBATCH --gres=gpu:1  
#SBATCH --mem=120G
#SBATCH --qos=benchmark

module load cuda/cuda_12.2/12.2

gpu=$1

executable="./cublas_f32"
output=../../result_data/cublas_${gpu}.txt
echo "output:$output"

M=(256 512 1024 2048 4096)
N=(2048 6144 2048 5632 4096 12288 4096 11008 5120 15360 5120 13824 6656 19968 6656 17920 8192 24576 8192 22016)
K=(2048 2048 5632 2048 4096 4096 11008 4096 5120 5120 13824 5120 6656 6656 17920 6656 8192 8192 22016 8192)

tmp=`mktemp`

for m in "${M[@]}"; do
    for ((i=0;i<${#N[@]};i++)) 
    do 
        n="${N[i]}"
        k="${K[i]}"
        $executable $m $n $k | tee $tmp
        cat $tmp |grep TFLOPS|cut -d' ' -f 5 >> $output
        echo ""
    done
done
