#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1   
#SBATCH --gres=gpu:1  
#SBATCH --qos=benchmark 

module load cuda/cuda_12.2/12.2

executable="./test_nmspmm"
raw_output="./output.txt"
>$raw_output

# M=(256 512 1024 2048 4096)
# N=(2048 6144 2048 5632 4096 12288 4096 11008 5120 15360 5120 13824 6656 19968 6656 17920 8192 24576 8192 22016)
# K=(2048 2048 5632 2048 4096 4096 11008 4096 5120 5120 13824 5120 6656 6656 17920 6656 8192 8192 22016 8192)
sparsity_list=(0.5 0.625 0.75 0.875)
M=(512 512  512  1024 2048 4096)
N=(512 1024 2048 2048 4096 4096)
K=(512 1024 2048 2048 4096 4096)
type_list=(1 2 3 4)
tmp=`mktemp`


for sparsity in "${sparsity_list[@]}"; do
    echo "***************************** sparsity = $sparsity *****************************"
    for ((i=0;i<${#M[@]};i++)) 
    do 
        max_tflops=0
        for type in "${type_list[@]}"; do
            m="${M[i]}"
            n="${N[i]}"
            k="${K[i]}"
            $executable $m $n $k 32 $sparsity $type 100 100 | tee $tmp
            cat $tmp >> $raw_output
        done
    done
done


calculate_tflops() {
  local pattern="$1"  # 获取输入参数
  local tflops=$(cat "$raw_output" | grep "$pattern" -A 9 | grep MAX | cut -d' ' -f2)
  
  local count=0
  local peaks=(9.836548 14.106404 17.521530 17.987152 18.644147 18.919600)  # 根据需要设置 scale 的值
  local scales=(2 2.666667 4 8)   # 根据需要设置 peak 的值

  # 遍历每一行
  echo "$tflops" | while IFS= read -r line; do
    scale_index=$((count / 6))  # 根据行号获取 scale 的索引
    peak_index=$((count % 6))    # 根据行号获取 peak 的索引

    scale=${scales[$scale_index]}  # 获取当前的 scale 值
    peak=${peaks[$peak_index]}      # 获取当前的 peak 值

    # 计算
    echo "scale=6; 100 * $line / $peak / $scale" | bc

    count=$((count + 1))  # 行计数器加1
  done

  echo -e "\n\n"
}

# Kernel for small matrices
calculate_tflops "32x32"

# Kernel for medium matrices
calculate_tflops "32x64"

# Kernel for large matrices
calculate_tflops "64x64"

# Kernel for huge matrices
calculate_tflops "64x128"