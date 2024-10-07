#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1   
#SBATCH --gres=gpu:1  
#SBATCH --qos=benchmark 

module load cuda/cuda_12.2/12.2

executable="./test"
output="search_out.txt"
>$output

# M=(256 512 1024 2048 4096)
# N=(2048 6144 2048 5632 4096 12288 4096 11008 5120 15360 5120 13824 6656 19968 6656 17920 8192 24576 8192 22016)
# K=(2048 2048 5632 2048 4096 4096 11008 4096 5120 5120 13824 5120 6656 6656 17920 6656 8192 8192 22016 8192)
sparsity_list=(0.5 0.625 0.75 0.875)
M=(128 256 512 1024 2048 4096)
N=(512 1024 1536 2048 2560 3072 3584 4096)
K=(512 1024 1536 2048 2560 3072 3584 4096)
type_list=(1 2 3 4)
splitk_list=(1 2 4 8)
tmp=`mktemp`


for sparsity in "${sparsity_list[@]}"; do
    echo "***************************** sparsity = $sparsity *****************************"
    for type in "${type_list[@]}"; do
        echo "###################### type = $type ######################"
        for sk in "${splitk_list[@]}"; do
            echo "================= splitk = $sk ================="
            for m in "${M[@]}"; do
                for ((i=0;i<${#N[@]};i++)) 
                do 
                    n="${N[i]}"
                    k="${K[i]}"
                    $executable $m $n $k 32 $sparsity $type $sk 100 100 | tee $tmp
                    tflops=`cat $tmp | grep MAX_TFLOPS| cut -d ' ' -f2`
                    echo $tflops >> $output
                done
            done
            echo -e "\n\n" >> $output
        done
    done
done
