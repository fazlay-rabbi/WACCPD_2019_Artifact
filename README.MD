### Overview
This zip file contains the sourcecodes of LOBPCG running on GPU and our tiled SpMM, XY and XTY kernels. 
Description of the contents of each subfolder are as follows:

### LOPBCG-GPU
This folder contains the following 4 subfolder:
  1. lib_csr_openacc_gpu
  2. lib_csr_openacc_cpu
  3. lib_csr_openmp_gpu
  4. lib_csr_openmp_cpu

First two subfolders contain cpu and gpu implementations of LOBPCG using OpenACC and the last two subfolders contain cpu and gpu implementations of LOBPCG using OpenMP.
Inside each folder there are makefiles to compile the respective program both on Summit and NERSC CORI-GPU supercomputer.

## SpMM-mini-app
This folder contains sourcecode of our tiled SpMM kernel. It also contains makefiles to compile it both on Summit and NERSC CORI-GPU supercomputer.

## cublasDgemm-mini-app

This folder contains sourcecode of our tiled linear combination kernel (XY) and inner product kernel (XTY). It also contains makefiles to compile the programs both on Summit and NERSC CORI-GPU supercomputer.

## Paper
This folder contains our WACCPD 2019 paper and presentation.
