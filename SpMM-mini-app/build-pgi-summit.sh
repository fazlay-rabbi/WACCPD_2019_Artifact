#!/bin/bash
module list
module load pgi/19.5
module load cuda
module list
set -x
set -e
export CUDA_ROOT=$CUDA_DIR
make clean
make
