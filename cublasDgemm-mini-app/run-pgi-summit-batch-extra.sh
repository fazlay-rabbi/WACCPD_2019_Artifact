#!/bin/bash
#BSUB -P GEN007port
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -J dgemm-parmat-extra
#BSUB -o dgemm-parmat-extra.%J

module list
module load pgi/19.5
module load cuda
module list

N=67108864
tile=131072

for M in 64; do
    for exe in xy.x xy_um.x; do
	jsrun -n 1 -a 1 -g 1 -c 1 \
	    --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" \
	    time \
	    ./exe/${exe} ${N} ${M} ${M} ${tile}
    done
done
