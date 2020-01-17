#!/bin/bash
module list
module load pgi/19.5
export PGI_LOCALRC=/usr/common/software/pgi/19.7/linux86-64/19.7/bin/localrc
module use /global/common/cori/software/modulefiles
module load cuda/10.0
module list

set -x
set -e
#export CUDA_ROOT=$CUDA_DIR
make clean
make
