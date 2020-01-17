CC=pgc++
MKL_ROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl
CPPFLAGS_pgi=-acc -ta=multicore -std=c++11 -pgf90libs -L/usr/common/software/pgi/$PGI_VERSION/linux86-64/$PGI_VERSION/include
CPPFLAGS=-acc -ta=multicore -std=c++11 -m64 -pgf90libs -I${MKLROOT}/include
EXEDIR=exe
LINKER_INT32=-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
LINKER_INT32_STATIC=-Wl,--start-group $(MKL_ROOT)/lib/intel64/libmkl_intel_lp64.a $(MKL_ROOT)/lib/intel64/libmkl_gnu_thread.a $(MKL_ROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

LINKER_DYNAMIC=-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
LINKER_DYNAMIC_pgi=-L/usr/common/software/pgi/$PGI_VERSION/linux86-64/$PGI_VERSION/lib -llapack -lblas  

#-lpgf90 -lpgf90_rpm1 -lpgf902 -lpgf90rtl -lpgftnrtl -lnspgc -lpgc -lrt -lm -lgcc -lc -lgcc ==> did the trick
#https://www.pgroup.com/userforum/viewtopic.php?p=7847

libcsr_v1:
	$(CC) $(CPPFLAGS_pgi) -c lib_csr_openacc.cpp -o $(EXEDIR)/lib_csr_openacc_v1.o $(LINKER_DYNAMIC_pgi)
	$(CC) $(CPPFLAGS_pgi) -c lobpcg_libcsr_vtune_v2_openacc.cpp -o $(EXEDIR)/lobpcg_libcsr_v1_openacc.o $(LINKER_DYNAMIC_pgi)
	$(CC) $(CPPFLAGS_pgi) $(EXEDIR)/lib_csr_openacc_v1.o $(EXEDIR)/lobpcg_libcsr_v1_openacc.o -o $(EXEDIR)/lobpcg_libcsr_v1_openacc.x $(LINKER_DYNAMIC_pgi)

clean:
	rm $(EXEDIR)/*.x $(EXEDIR)/*.o
