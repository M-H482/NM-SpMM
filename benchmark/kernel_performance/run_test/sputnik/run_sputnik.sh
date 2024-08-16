#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1   
#SBATCH --gres=gpu:1  

source env.sh
module load cuda/cuda_12.2/12.2

gpu=$1
sparsity=$2

n_iter=100
executable="./test_sputnik"
output=../../result_data/sputnik_${gpu}_${sparsity}.txt
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
        $executable $sparsity $m $n $k $n_iter | tee $tmp
        cat $tmp | grep TFLOPS |cut -d' ' -f3 |cut -d',' -f1 >> $output
        echo ""
    done
done
