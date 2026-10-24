#!/bin/bash
# Compile this candidate's kernel.cu into libkda.so for sm_89.
set -e
export PATH="/usr/local/cuda-12.8/bin:$PATH"
cd "$(dirname "${BASH_SOURCE[0]}")"
nvcc -arch=sm_89 -O3 --shared -Xcompiler -fPIC -o libkda.so kernel.cu
echo "compiled libkda.so"
