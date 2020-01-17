#!/bin/bash
module list
module load xl
module load cuda
module load essl
module load netlib-lapack
module list
set -x
set -e
export CUDA_ROOT=$CUDA_DIR
make -f Makefile.xl clean
make -f Makefile.xl
