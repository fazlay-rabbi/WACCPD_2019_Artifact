#!/bin/bash
#BSUB -P GEN007port
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags NVME
#BSUB -J spmm-Nm7
#BSUB -o spmm-Nm7.%J

module list
module load pgi/19.5
module load cuda
module list

tile=2597152
matrix=Nm7_CSR.dat

# Copy the input matrix into the Burst Buffer
jsrun -n 1 cp /gpfs/alpine/csc190/scratch/cdaley/Matrices/${matrix} /mnt/bb/cdaley

for rhs in 8 12 16 24 32 48; do
    for exe in spmm.x spmm_um.x; do
	echo "Running ${exe} with rhs=${rhs} and tile size=${tile}"
	jsrun -n 1 -a 1 -g 1 -c 1 \
	    --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" \
	    time \
	    nvprof --openmp-profiling off \
	    ./exe/${exe} ${rhs} ${tile} /mnt/bb/cdaley/${matrix}
    done
done
