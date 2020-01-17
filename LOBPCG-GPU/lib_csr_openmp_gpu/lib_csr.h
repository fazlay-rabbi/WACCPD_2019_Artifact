#ifndef LIB_CSR_H
#define LIB_CSR_H
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
// #include <cmath>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <fstream>
using namespace std;

#if defined(USE_MKL)
#include <mkl.h>
#endif
#if defined(USE_LAPACK)
#include <lapacke.h>
#include <cblas.h>
#endif

#include <omp.h>

#ifdef USE_CUBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

extern long position;
extern int *colptrs, *irem;
extern int  *ia , *ja;

extern int numcols, numrows, nnonzero, nthrds;
extern int nrows, ncols, nnz;
extern int wblk, nrowblks, ncolblks;

//CSB structure
template<typename T>
struct block
{
    int nnz;
    int roffset, coffset;
    unsigned short int *rloc, *cloc;
    T *val;
};
void spmm_csr(int row, int col, int nvec, int *row_ptr, int *col_index, double *value, double *Y, double *result);
void _XTY(double *X, double *Y, double *result ,int M, int N, int P, int blocksize);
void transpose(double *src, double *dst, const int N, const int M);
void inverse(double *arr, int m, int n);
void print_mat(double *arr, const int row, const int col);
void make_identity_mat(double *arr, const int row, const int col);
void diag(double *src, double *dst, const int size);
void mat_sub(double *src1, double *src2, double *dst, const int row, const int col);
void mat_addition(double *src1, double *src2, double *dst, const int row, const int col);
void mat_mult(double *src1, double *src2, double *dst, const int row, const int col);
void sum_sqrt(double *src, double *dst, const int row, const int col);
void update_activeMask(int *activeMask, double *residualNorms, double residualTolerance, int blocksize);
void getActiveBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void updateBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void mat_copy(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst);
void print_eigenvalues( int n, double* wr, double* wi );

template<typename T>
void read_custom(char* filename, T *&xrem);

// template<typename T>
// void read_custom_csr(char* filename, T *&xrem);

template<typename T>
void csc2blkcoord(block<T> *&matrixBlock, T *xrem);

// template<typename T>
// void spmm_blkcoord(int R, int C, int M, int nthrds, T *X,  T *Y, block<T> *H);

// template<typename T>
// void read_ascii(char* filename, T *&xrem);

void custom_dlacpy(double *src, double *dst, int m, int n);

template<typename T>
bool checkEquals( T* a, T* b, size_t outterSize, size_t innerSize);


///============= GPU version ================
void custom_dlacpy_GPU(double *src, double *dst, int m, int n);
void mat_addition_GPU(double *src1, double *src2, double *dst, const int row, const int col);
void mat_sub_GPU(double *src1, double *src2, double *dst, const int row, const int col);
void mat_sub_GPU_v2(double *src1, double *src2, double *dst, const int row, const int col);
void mat_mult_GPU(double *src1, double *src2, double *dst, const int row, const int col);
void mat_mult_GPU_v2(double *src1, double *src2, double *dst, const int row, const int col);
void sum_sqrt_GPU(double *src, double *dst, const int row, const int col);
void sum_sqrt_GPU_v2(double *src, double *dst, const int row, const int col);
void update_activeMask_GPU(int *activeMask, double *residualNorms, double residualTolerance, int blocksize);
void getActiveBlockVector_GPU(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void getActiveBlockVector_GPU_v2(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void updateBlockVector_GPU(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void mat_copy_GPU(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst);
void transpose_GPU(double *src, double *dst, const int N, const int M);
void make_identity_mat_GPU(double *arr, const int row, const int col);

#if defined(USE_OPENACC)
void mat_addition_OpenACC(double *src1, double *src2, double *dst, const int row, const int col);
void mat_sub_OpenACC(double *src1, double *src2, double *dst, const int row, const int col);
void mat_mult_OpenACC(double *src1, double *src2, double *dst, const int row, const int col);
void getActiveBlockVector_OpenACC(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void sum_sqrt_OpenACC(double *src, double *dst, const int row, const int col);
void update_activeMask_OpenACC(int *activeMask, double *residualNorms, double residualTolerance, int blocksize);
void custom_dlacpy_OpenACC(double *src, double *dst, int m, int n);
void updateBlockVector_OpenACC(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void transpose_OpenACC(double *src, double *dst, const int N, const int M);
void mat_copy_OpenACC(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst);
#endif

#endif
