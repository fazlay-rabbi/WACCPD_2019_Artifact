#!/bin/bash
#BSUB -P GEN007port
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags NVME
#BSUB -J lobpcg-xl
#BSUB -o lobpcg-xl.%J

module list
module load xl
module load cuda
module load essl
module load netlib-lapack
module list

set -x
for matrix in Queen_4147_CSR.dat HV15R_CSR.dat Nm7_CSR.dat Nm8_CSR.dat; do
    # Copy the input matrix into the Burst Buffer
    jsrun -n 1 cp /gpfs/alpine/csc190/scratch/cdaley/Matrices/${matrix} /mnt/bb/cdaley

    for rhs in 4 8 12 16; do
	echo "Running ${exe} with rhs=${rhs} and tile size=${tile}"
	jsrun -n 1 -a 1 -g 1 -c 1 \
	    time \
	    ./exe/lobpcg_libcsr_GPU_v1_openmp.x ${rhs} /mnt/bb/cdaley/${matrix}
    done
done

echo "Using NVPROF"
for matrix in Queen_4147_CSR.dat HV15R_CSR.dat Nm7_CSR.dat Nm8_CSR.dat; do
    for rhs in 4 8 12 16; do
	jsrun -n 1 -a 1 -g 1 -c 1 \
	    --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" \
	    time \
	    nvprof --openmp-profiling off \
	    ./exe/lobpcg_libcsr_GPU_v1_openmp.x ${rhs} /mnt/bb/cdaley/${matrix}
    done
done
