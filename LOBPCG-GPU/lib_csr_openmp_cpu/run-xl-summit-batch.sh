#!/bin/bash
#BSUB -P GEN007port
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags NVME
#BSUB -J lobpcg-xl-cpu
#BSUB -o lobpcg-xl-cpu.%J

module list
module load xl
module load essl
module load netlib-lapack
module list

set -x
export OMP_NUM_THREADS=21

for matrix in Queen_4147_CSR.dat HV15R_CSR.dat Nm7_CSR.dat Nm8_CSR.dat; do
    # Copy the input matrix into the Burst Buffer
    jsrun -n 1 cp /gpfs/alpine/csc190/scratch/cdaley/Matrices/${matrix} /mnt/bb/cdaley

    for rhs in 4 8 12 16; do
	echo "Running ${exe} with rhs=${rhs} and tile size=${tile}"
	jsrun -n 1 -a 1 -c 42 --bind=packed:21 \
	    time \
	    ./exe/lobpcg_libcsr_vtune_v1_xl.x ${rhs} /mnt/bb/cdaley/${matrix}
    done
done
