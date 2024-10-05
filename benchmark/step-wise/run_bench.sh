#!/bin/bash

# default on A100
arch="sm_80"

# Parse parameters
while getopts "a:" opt; do
  case $opt in
    a) arch="$OPTARG" ;;  # Set arch to the provided parameter value
    *) echo "Usage: $0 [-a architecture]" >&2
       exit 1 ;;
  esac
done

# Select the measured TFLOPS for cuBLAS FP32 dense GEMM.
case $arch in
  sm_80)
    peak=19
    ;;
  sm_86)
    peak=24.5
    ;;
  sm_89)
    peak=56
    ;;
  *)
    echo "Unsupported architecture: $arch" >&2
    exit 1
    ;;
esac

bash build.sh -a $arch
echo "The corresponding peak value is: $peak"

exe_v0_=./v0_naive
exe_v1_=./v1_block_tiling
exe_v2_=./v2_warp_thread_tiling
exe_v3L=./v3_sparsity_aware_low_sparsity
exe_v3H=./v3_sparsity_aware_high_sparsity
exe_v4L=./v4_prefetch_low_sparsity 
exe_v4H=./v4_prefetch_high_sparsity 

m=4096; n=4096; k=4096; 
echo "sparsity = 0%"
s=0; scale=1;
tflops=$($exe_v0_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v1_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v2_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v3L  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v4L  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc

echo "sparsity = 50.0%"
s=0.5; scale=2;
tflops=$($exe_v0_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v1_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v2_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v3L  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v4L  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc

echo "sparsity = 62.5%"
s=0.625; scale=2.666666;
tflops=$($exe_v0_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v1_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v2_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v3L  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v4L  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc

echo "sparsity = 75%"
s=0.75; scale=4;
tflops=$($exe_v0_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v1_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v2_  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v3H  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v4H  $m $n $k 32 $s 100 100 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc

echo "sparsity = 87.5%"
s=0.875; scale=8;
tflops=$($exe_v0_  $m $n $k 32 $s 300 300 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v1_  $m $n $k 32 $s 300 300 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v2_  $m $n $k 32 $s 300 300 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v3H  $m $n $k 32 $s 300 300 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
tflops=$($exe_v4H  $m $n $k 32 $s 300 300 | grep TFLOPS | cut -d ' ' -f 5)
echo "scale=6; 100 * $tflops / $peak / $scale" | bc
