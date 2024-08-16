#!/bin/bash

exe_v0_=./v0_naive
exe_v1_=./v1_block_tiling
exe_v2_=./v2_warp_thread_tiling
exe_v3L=./v3_sparsity_aware_low_sparsity
exe_v3H=./v3_sparsity_aware_high_sparsity
exe_v4L=./v4_prefetch_low_sparsity 
exe_v4H=./v4_prefetch_high_sparsity 

echo "sparsity = 50.0%"
s=0.5
m=4096; n=4096; k=4096; 
$exe_v0_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v1_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v2_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v3L  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v4L  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 

echo "sparsity = 62.5%"
s=0.625
m=4096; n=4096; k=4096; 
$exe_v0_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v1_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v2_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v3L  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v4L  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 

echo "sparsity = 75%"
s=0.75
m=4096; n=4096; k=4096; 
$exe_v0_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v1_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v2_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v3H  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v4H  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 

echo "sparsity = 87.5%"
s=0.875
m=4096; n=4096; k=4096; 
$exe_v0_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v1_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v2_  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v3H  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 
$exe_v4H  $m $n $k 32 $s | grep TFLOPS | cut -d ' ' -f 5 