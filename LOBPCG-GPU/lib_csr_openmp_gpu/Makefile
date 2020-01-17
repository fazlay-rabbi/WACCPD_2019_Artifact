CC=clang++ -DUSE_LAPACK
CPPFLAGS=-fopenmp -O3 -std=c++11 -fopenmp-targets=nvptx64-nvidia-cuda
EXEDIR=exe
INC= -I/global/homes/r/rabbimd/lapack-3.8.0/LAPACKE/include -I/global/homes/r/rabbimd/CBLAS/include
LIB=-L/global/homes/r/rabbimd/lapack-3.8.0/lib -L/global/homes/r/rabbimd/CBLAS/lib -llapacke -llapack -lcblas -lblas -lgfortran -lm

MKL_ROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
EXEDIR=exe

CUFLAGS=-DUSE_CUBLAS -DUSE_OPENACC
CFLAGS=-I${CUDA_ROOT}/include
LIBS=-L${CUDA_ROOT}/lib64 -lcublas -lcudart -lcusparse

gpu_v1:
	$(CC) $(CPPFLAGS) $(CUFLAGS) $(CFLAGS) $(LIBS) $(INC) -c lib_csr.cpp -o $(EXEDIR)/lib_csr_openmp.o $(LIB)
	$(CC) $(CPPFLAGS) $(CUFLAGS) $(CFLAGS) $(LIBS) $(INC) -c lobpcg_libcsr_GPU_v1_openmp.cpp -o $(EXEDIR)/lobpcg_libcsr_GPU_v1_openmp.o $(LIB)
	$(CC) $(CPPFLAGS) $(CUFLAGS) $(CFLAGS) $(LIBS) $(INC) $(EXEDIR)/lib_csr_openmp.o $(EXEDIR)/lobpcg_libcsr_GPU_v1_openmp.o -o $(EXEDIR)/lobpcg_libcsr_GPU_v1_openmp.x $(LIB)
clean:
	rm $(EXEDIR)/*.x $(EXEDIR)/*.o 