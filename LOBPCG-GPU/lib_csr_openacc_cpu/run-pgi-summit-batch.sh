#!/bin/bash
#BSUB -P GEN007port
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags NVME
#BSUB -J lobpcg-pgi-cpu
#BSUB -o lobpcg-pgi-cpu.%J

module list
module load pgi/19.5
module list

set -x
export OMP_NUM_THREADS=21

for matrix in Queen_4147_CSR.dat HV15R_CSR.dat Nm7_CSR.dat Nm8_CSR.dat; do
    # Copy the input matrix into the Burst Buffer
    jsrun -n 1 -a 1 -c 42 --bind=packed:21 \
	cp /gpfs/alpine/csc190/scratch/cdaley/Matrices/${matrix} /mnt/bb/cdaley

    for rhs in 4 8 12 16; do
	echo "Running ${exe} with rhs=${rhs} and tile size=${tile}"
	jsrun -n 1 -a 1 -c 42 --bind=packed:21 \
	    time \
	    ./exe/lobpcg_libcsr_v1_openacc.x ${rhs} /mnt/bb/cdaley/${matrix}
    done
done
