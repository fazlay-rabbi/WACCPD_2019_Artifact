#!/bin/bash
module list
module load xl
module load essl
module load netlib-lapack
module list
set -x
set -e
make -f makefile_xl clean
make -f makefile_xl
