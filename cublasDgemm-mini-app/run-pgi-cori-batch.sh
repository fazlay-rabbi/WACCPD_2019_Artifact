#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -A nstaff
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -c 10

module list
module load pgi/19.5
module load cuda
module list

set -x
make clean
make

N=67108864
tile=131072

export OMP_NUM_THREADS=1
for M in 16 24 32 48 64; do
    for exe in xty.x xty_um.x; do
	srun -n 1 -c 10 --cpu-bind=cores \
	    time \
	    nvprof --openmp-profiling off \
	    --unified-memory-profiling per-process-device \
	    ./exe/${exe} ${N} ${M} ${tile}
    done
done

for M in 16 24 32 48 64; do
    for exe in xy.x xy_um.x; do
	srun -n 1 -c 10 --cpu-bind=cores \
	    time \
	    nvprof --openmp-profiling off \
	    --unified-memory-profiling per-process-device \
	    ./exe/${exe} ${N} ${M} ${M} ${tile}
    done
done
