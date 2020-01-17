#!/bin/bash -l

#SBATCH -A m1759
#SBATCH -N 1
#SBATCH -t 1:00:00  
#SBATCH -q regular   
#SBATCH -C gpu
#SBATCH -c 80
#SBATCH --gres=gpu:8

module list
module load pgi/19.5
export PGI_LOCALRC=/usr/common/software/pgi/19.7/linux86-64/19.7/bin/localrc
module use /global/common/cori/software/modulefiles
module load cuda/10.0
module list
export ACC_NUM_CORES=20

tile=4985422
matrix=Nm7_CSR.dat


for rhs in 8 12 16 24 32 48; do
    for exe in spmm.x spmm_um.x; do
	echo "Running ${exe} with rhs=${rhs} and tile size=${tile}"
	srun -n 1 -c 40 --cpu-bind=cores \
	    time \
	    nvprof \
	    ./exe/${exe} ${rhs} ${tile} /global/cscratch1/sd/rabbimd/DeepSparse_NERSC_COLAB/Matrices/${matrix}
    done
done
