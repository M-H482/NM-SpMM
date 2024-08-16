#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1   
#SBATCH --gres=gpu:1  
#SBATCH --qos=benchmark 

module load cuda/cuda_12.2/12.2

gpu=$1
sparsity=$2

executable="./test_nmspmm"
output=../../result_data/nmspmm_${gpu}_${sparsity}.txt
echo "output:$output"

M=(256 512 1024 2048 4096)
N=(2048 6144 2048 5632 4096 12288 4096 11008 5120 15360 5120 13824 6656 19968 6656 17920 8192 24576 8192 22016)
K=(2048 2048 5632 2048 4096 4096 11008 4096 5120 5120 13824 5120 6656 6656 17920 6656 8192 8192 22016 8192)
spliK_list=(1 2 4 8)
tmp=`mktemp`

for m in "${M[@]}"; do
    for ((i=0;i<${#N[@]};i++)) 
    do 
        max_tflops=0
        for spliK in "${spliK_list[@]}"; do
            n="${N[i]}"
            k="${K[i]}"
            $executable $m $n $k 32 $sparsity $spliK | tee $tmp
            tflops=`cat $tmp | grep MAX_TFLOPS| cut -d ' ' -f2`
            echo ""
            if (( $(echo "scale=6; $tflops > $max_tflops" | bc -l) )); then
                max_tflops=$tflops
            fi
        done
        echo $max_tflops >> $output
    done
done
