#!/bin/bash

# Load user environment (optional)
source ~/.bashrc

# Remove old binary
rm -f benchmark_scan

# Compile with nvcc for RTX 4060 (Ada Lovelace, CC 8.9)
nvcc -O3 -arch=sm_89 -Xcompiler -fopenmp benchmark_scan_cuda.cu -o benchmark_scan_cuda

# Run the benchmark
./benchmark_scan_cuda
