CFLAGS_CPU=-acc -O3 -DUSE_CPU -ta=multicore

CFLAGS_MKL=-fopenmp -O3 -std=c++11  -m64 -I${MKL_ROOT}/include
CFLAGS_UM=-acc -O3 -DUSE_UM -ta:tesla:managed -ta=tesla:cc70,cuda10.0

MKL_ROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
LINKER_DYNAMIC= -L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

EXE=exe

all: spmm spmm_um

spmm:
	pgc++ -std=c++11 -acc -w -O3 -DUSE_GPU -ta=tesla:cc70,cuda10.0 -L$(CUDA_ROOT)/lib64 -lcublas -lcudart -lcusparse spmm_mini_app.cpp -o $(EXE)/spmm.x
spmm_um:
	pgc++ -std=c++11 $(CFLAGS_UM) -DUSE_UM -L$(CUDA_ROOT)/lib64 -w -lcublas -lcudart -lcusparse spmm_mini_app.cpp -o $(EXE)/spmm_um.x

clean:
	rm -f $(EXE)/spmm.x $(EXE)/spmm_um.x
