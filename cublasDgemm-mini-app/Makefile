
#CC=cc -DUSE_NVBLAS
CC=pgcc
LINKER_DYNAMIC_pgi=-L/usr/common/software/pgi/$PGI_VERSION/linux86-64/$PGI_VERSION/lib -llapack -lblas 
EXE=exe

CFLAGS_UM=-acc -O3 -DUSE_UM -ta:tesla:managed -ta=tesla:cc70,cuda10.0 
CFLAGS=-acc -O3 -DUSE_CUBLAS -ta=tesla:cc70,cuda10.0 
CFLAGS_MKL=-acc -O3 -DUSE_MKL -ta=multicore $(LINKER_DYNAMIC_pgi) -pgf90libs -L/usr/common/software/pgi/$PGI_VERSION/linux86-64/$PGI_VERSION/include
CFLAGS_TILE=-acc -O3 -DUSE_CUBLAS -ta=tesla:cc70,cuda10.0 
LIBS=-L$(CUDA_ROOT)/lib64 -lcublas -lcudart

all: xy xy_um xty xty_um

xy:
	$(CC) $(CFLAGS_TILE) -Minfo -o $(EXE)/xy.x cublasDgemm_xy.c $(LIBS)
xy_um:
	$(CC) $(CFLAGS_UM) -Minfo -o $(EXE)/xy_um.x cublasDgemm_xy.c $(LIBS)
xty:
	$(CC) $(CFLAGS_TILE) -Minfo -o $(EXE)/xty.x cublasDgemm_xty.c $(LIBS)
xty_um:
	$(CC) $(CFLAGS_UM) -Minfo -o $(EXE)/xty_um.x cublasDgemm_xty.c $(LIBS)
dgemm-tiled-xty-MKL:
	$(CC) $(CFLAGS_MKL) -Minfo -o $(EXE)/dgemm-tiled-xty-MKL.x mt-dgemm-tiled_xty.c $(LIBS)
clean:
	rm -f $(EXE)/*.x

