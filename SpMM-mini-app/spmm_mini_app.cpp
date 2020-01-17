#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

#include <sys/time.h>
#if defined(USE_GPU) || defined(CHECK_SANITY) || defined(USE_UM)
#include <cuda_runtime.h>
#include "cusparse.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#endif

#include <openacc.h>

#if defined(USE_MKL)
#include <mkl.h>
#endif

#if defined(USE_OPENMP)
#include <omp.h>
#endif

//https://stackoverflow.com/questions/51345922/number-of-operations-for-sparsedense-matrix-multiplication

#if defined(USE_GPU) || defined(USE_UM)
void transpose_GPU(double *src, double *dst, const int N, const int M)
{
    //src - M * N
    int i, j;
    #pragma acc parallel loop deviceptr(src, dst)
    for(i = 0 ; i < M ; i++)
    {
        #pragma acc loop
        for(j = 0 ; j < N ; j++)
        {
            dst[j * M + i] = src[i * N + j];
        }
    }
}
#endif

double get_seconds() {
	struct timeval now;
	gettimeofday(&now, NULL);

	const double seconds = (double) now.tv_sec;
	const double usec    = (double) now.tv_usec;

	return seconds + (usec * 1.0e-6);
}

void transpose_CPU(double *src, double *dst, const int N, const int M)
{
    //src - M * N
    int i, j;
    // #pragma acc parallel loop copyin(src[0: N*M]) copyout(dst[0: N*M])
    // #pragma acc kernels loop
    // #pragma acc enter data copyin(src[0: N*M], dst[0: N*M])
    // #pragma acc update self(dst[0: N*M])
    // #pragma acc parallel loop
    for(i = 0 ; i < M ; i++)
    {
        // #pragma acc loop
        for(j = 0 ; j < N ; j++)
        {
            dst[j * M + i] = src[i * N + j];
        }
    }
    // #pragma acc exit data delete( src[0 : N * M], dst[0 : N * M])
}

#if defined(USE_CPU)
void spmm_csr(int row, int col, int nvec, int *row_ptr, int *col_index, double *value, double *Y, double *result)
{
    int i, j, k, start, end;
    int r, c, xcoef;

    //#pragma omp parallel for default(shared) private(start, end, r, c, xcoef, j, k)
    #pragma acc parallel loop private(start, end, r, c, xcoef, j, k)
    for(i = 0 ; i < row ; i++)
    {
        start = row_ptr[i];
        end = row_ptr[i + 1];
        //printf("%d %d %d\n", i, start, end);
        //printf("col_index[%d]: %d col[%d]: %d\n", start, col_index[start], end, col_index[end]);
        for(j = start ; j < end ; j++)
        {
            r = i;
            c = col_index[j];
            xcoef = value[j];  
            //printf("row: %d col: %d value: %lf\n", r, c, xcoef);
            #pragma acc loop
            for(k = 0 ; k < nvec ; k++)
            {
                result[r * nvec + k] = result[r * nvec + k] + xcoef * Y[c * nvec + k];
            }
        }
    }
}
#endif

#if defined(USE_OPENMP)
void spmm_csr_OpenMP(int row, int col, int nvec, int *row_ptr, int *col_index, double *value, double *Y, double *result)
{
    int i, j, k, start, end;
    int r, c, xcoef;

    #pragma omp parallel for private(start, end, r, c, xcoef, j, k)
    // #pragma acc parallel loop //private(start, end, r, c, xcoef, j, k)
    for(i = 0 ; i < row ; i++)
    {
        start = row_ptr[i];
        end = row_ptr[i + 1];
        //printf("%d %d %d\n", i, start, end);
        //printf("col_index[%d]: %d col[%d]: %d\n", start, col_index[start], end, col_index[end]);
        for(j = start ; j < end ; j++)
        {
            r = i;
            c = col_index[j];
            xcoef = value[j];  
            //printf("row: %d col: %d value: %lf\n", r, c, xcoef);
            // #pragma acc loop
            for(k = 0 ; k < nvec ; k++)
            {
                result[r * nvec + k] = result[r * nvec + k] + xcoef * Y[c * nvec + k];
            }
        }
    }
}
#endif

int main(int argc, char *argv[])
{
    int M, N, index = 0, currentBlockSize, block_width, blksz;
    double dzero = 0.0, dtwo = 2.0, dthree = 3.0, done = 1.0;
    int i, j, k;
    currentBlockSize = atoi(argv[1]);
    block_width = atoi(argv[2]);
    double gpu_memory, matrix_memory;

    double alpha = 1.0, beta = 0.0;

    int *ia, *ja, nnonzero, numrows, numcols, nthrds, wblk;
    double *acsr;
    // char *filename = argv[3] ; 
    wblk = block_width; 

    // *--------- Reading CSR from binary file ---------* //
    char *filename = argv[3];
    ifstream file (filename, ios::in|ios::binary);
    if (file.is_open())
    {
        file.read ((char*)&numrows,sizeof(numrows));
        // cout << "row: "<<numrows<<endl;
        
        file.read(reinterpret_cast<char*>(&numcols), sizeof(numcols));
        // cout << "colum: " << numcols << endl;

        file.read(reinterpret_cast<char*>(&nnonzero), sizeof(nnonzero));
        // cout << "non zero: " << nnonzero << endl;

        ia = (int *) malloc((numrows + 1) * sizeof(int)); //colsptr
        ja = (int *) malloc(nnonzero * sizeof(int)); //irem
        acsr = (double *) malloc(nnonzero * sizeof(double)); //xrem

        i = 0;
        while(!file.eof() && i <= numrows)
        {
            file.read(reinterpret_cast<char*>(&j), sizeof(j)); //irem(j)
            ia[i++] = j;
        }
        // cout << "finished reading ia"<<endl;
        i = 0;
        while(!file.eof() && i < nnonzero)
        {
            file.read(reinterpret_cast<char*>(&j), sizeof(j)); //irem(j)
            ja[i++] = j;
        }
        // cout << "finished reading ja"<<endl;
        i = 0;
        double d;
        while(!file.eof() && i < nnonzero)
        {
            file.read(reinterpret_cast<char*>(&d), sizeof(d)); //irem(j)
            acsr[i++] = d;
        }  
        // cout << "finished reading acsr"<<endl;
    }
    file.close();
    // cout << "ia[0]: " << ia[0] << " ia[last]: " << ia[numrows] << endl;
    // cout << "ja[0]: " << ja[0] << " ja[last]: " << ja[nnonzero-1] << endl;
    // cout << "acsr[0]: " << acsr[0] << " acsr[last]: " << acsr[nnonzero-1] << endl;
    
    printf("          # of RHS vector =  %d\n", currentBlockSize);
#if !defined(USE_UM)
    printf("                Tile size =  %d\n", block_width);
#endif
    printf("                  numrows =  %d\n", numrows);
    printf("                  numcols =  %d\n", numcols);
    printf("                 nnonzero =  %d\n", nnonzero);

    double sparse_mtx_total = nnonzero * sizeof(double) * 1e-9 + nnonzero * sizeof(int) * 1e-9 + (numrows+1) * 4 * sizeof(int) * 1e-9;
    double dense_mtx_total = 2 * numrows * currentBlockSize * sizeof(double) * 1e-9;
    matrix_memory = sparse_mtx_total + dense_mtx_total;
    // *--------- Reading from txt file finished ---------* //
    M = numrows; N = numcols;

    double *d_activeBlockVectorR, *d_activeBlockVectorAR, *d_temp_actAR, *d_temp_actR;
    double *activeBlockVectorR, *activeBlockVectorAR, *temp_actR;
    
    activeBlockVectorR = (double *) malloc(numrows * currentBlockSize * sizeof(double));
    activeBlockVectorAR = (double *) malloc(numrows * currentBlockSize * sizeof(double));
    temp_actR = (double *) malloc(numrows * currentBlockSize * sizeof(double));

    if ((!activeBlockVectorR) || (!activeBlockVectorAR)){
        printf("Host malloc failed (matrix)");
        return 1;
    }

    srand(0);
    for(i = 0 ; i < numrows ; i++)
    {
        for(j = 0 ; j < currentBlockSize ; j++)
        {
            activeBlockVectorR[i * currentBlockSize + j] = 0.5 + (double)rand()/(double)RAND_MAX;
            activeBlockVectorAR[i * currentBlockSize + j] = 0.0;
        }
    }

#if defined(USE_GPU)
    // transpose_CPU(activeBlockVectorR, temp_actR, currentBlockSize, numrows);

    cudaError_t cudaStat, cudaStat1, cudaStat2;
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    /* initialize cusparse library */
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE Library initialization failed");
        return 1;
    }
    
    /* create and setup matrix descriptor */
    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Matrix descriptor initialization failed\n");
        return 1;
    }
    
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    /* exercise conversion routines (convert matrix from COO 2 CSR format) */
    cudaStat = cudaMalloc((void**)&d_activeBlockVectorR, numrows * currentBlockSize * sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf("Device malloc failed (d_activeBlockVectorR) errorcode: %d\n", cudaStat);
        return 1;
    }
    cudaStat = cudaMalloc((void**)&d_activeBlockVectorAR, block_width * currentBlockSize * sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf("Device malloc failed (d_activeBlockVectorAR) errorcode: %d\n", cudaStat);
        return 1;
    }
    cudaStat = cudaMalloc((void**)&d_temp_actAR, block_width * currentBlockSize * sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf("Device malloc failed (d_temp_actAR) errorcode: %d\n", cudaStat);
        return 1;
    }
     cudaStat = cudaMalloc((void**)&d_temp_actR, numrows * currentBlockSize * sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf("Device malloc failed (d_temp_actR) errorcode: %d\n", cudaStat);
        return 1;
    }
    // cudaStat = cudaMemcpy(d_activeBlockVectorR, temp_actR, numrows * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
    // if (cudaStat != cudaSuccess) {
    //     printf("cudaMemcpy failed (d_activeBlockVectorR) errorcode: %d\n", cudaStat);
    //     return 1;
    // }

    cudaStat = cudaMemcpy(d_temp_actR, activeBlockVectorR, numrows * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
        printf("cudaMemcpy failed (d_temp_actR) errorcode: %d\n", cudaStat);
        return 1;
    }
    
    transpose_GPU(d_temp_actR, d_activeBlockVectorR, currentBlockSize, numrows);
    cudaFree(d_temp_actR);

    int nrowblk = ceil(1.0 * numrows / block_width);
    int *nnz_per_tile = (int *)malloc(nrowblk * sizeof(int));
    int max_nnz = 0;
    
    printf("                  nrowblk =  %d\n", nrowblk);
    int totalnnz = 0;
    double time_taken = 0;
    for(i = 0 ; i < nrowblk; i++)
    {
        if(i < nrowblk - 1)
            nnz_per_tile[i] = ia[(i + 1) * block_width] - ia[i * block_width];
        else
            nnz_per_tile[i] = ia[numrows] - ia[i * block_width];

        if(max_nnz < nnz_per_tile[i])
            max_nnz = nnz_per_tile[i];
        
        totalnnz += nnz_per_tile[i];
        // printf("tile id: %d ==> nnz : %d %d : %d %d\n", i, nnz_per_tile[i], (i+1)*block_width, ia[(i+1)*block_width], ia[i*block_width]);
        // if(i == nrowblk - 2)
        //     printf("max_nnz: %d totalnnz: %d \n", max_nnz, totalnnz);

    }
    
    printf("            max_nnz/block =  %d\n", max_nnz);
    printf("                 totalnnz =  %d\n", totalnnz);

    printf("\n");
	printf("===============================================================\n");

    double max_sparse_block_size = max_nnz * sizeof(double) * 1e-9 + max_nnz * sizeof(int) * 1e-9 + (block_width+1) * 4 * sizeof(int) * 1e-9;
    double dense_mtx_size = numrows * currentBlockSize * sizeof(double) * 1e-9 + block_width * currentBlockSize * sizeof(double) * 1e-9;
    gpu_memory = max_sparse_block_size+dense_mtx_size;
    // printf("Max Sparse Matrix size: %lf GB\nDense Matrix size: %lf GB\n", max_sparse_block_size, dense_mtx_size);

    int *rowPtrTile, *colIndexTile;
    double *coolValTile;
    cudaStat = cudaMalloc((void**)&rowPtrTile, (block_width + 1) * sizeof(int));
    cudaStat1 = cudaMalloc((void**)&colIndexTile, max_nnz * sizeof(int));
    cudaStat2 = cudaMalloc((void**)&coolValTile, max_nnz * sizeof(double));
    if(cudaStat != 0 || cudaStat1 != 0 || cudaStat2 != 0)
    {
        printf("Failed to allocate rowPtrTile or colIndexTile\n"); 
        return 0;
    }

    for(i = 0 ; i < nrowblk; i++)
    {
        blksz = block_width;
        if(i * block_width + blksz > numrows)
            blksz = numrows - i * block_width;
        // printf("Tile id: %d blksz: %d\n", i, blksz);
        const double start = get_seconds();
        //now copy rowptr and colIndex of each block to GPU
        cudaStat = cudaMemcpy(rowPtrTile, ia+(i * block_width), (blksz + 1) * sizeof(int), cudaMemcpyHostToDevice);
        if( cudaStat != cudaSuccess ){ printf("cudaMemcpy failed rowPtrTile ==> %d\n", cudaStat1); return 1; }

        //     //copy colIndex corresponding to current block to GPU
        cudaStat = cudaMemcpy(colIndexTile, ja+ia[i * block_width], nnz_per_tile[i] * sizeof(int), cudaMemcpyHostToDevice);
        if( cudaStat != cudaSuccess ){ printf("cudaMemcpy failed colIndexTile ==> %d\n", cudaStat1); return 1; }
        // cudaDeviceSynchronize();
    //     //copy cooVal corresponding to current block to GPU
        cudaStat = cudaMemcpy(coolValTile, acsr+ia[i * block_width], nnz_per_tile[i] * sizeof(double), cudaMemcpyHostToDevice);
        if( cudaStat != cudaSuccess ){ printf("cudaMemcpy failed coolValTile ==> %d\n", cudaStat1); return 1; }

        cudaDeviceSynchronize();
        //offsetting rowptr
        #pragma acc kernels deviceptr(rowPtrTile)
        for(j = blksz; j >= 0 ; j--)
            rowPtrTile[j] = rowPtrTile[j] - rowPtrTile[0];

    //     //========== check rowptr =============
    //     cudaStat1 = cudaMemcpy(rowptrCheck, rowPtrTile, (block_width + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    //     if (cudaStat1 != cudaSuccess)  {
    //         printf("Memcpy from Device to Host failed");
    //         return 1;
    //     }
    //     for(int j = 0 ; j < block_width + 1 ; j++)
    //         printf("%d ", rowptrCheck[j]);
    //     printf("\n");
    //     //===================================

        
    
        // cudaDeviceSynchronize();
        
        status = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, blksz, currentBlockSize, numrows,
                           nnz_per_tile[i], &done, descr, coolValTile, rowPtrTile, colIndexTile,
                           d_activeBlockVectorR, numrows, &dzero, d_activeBlockVectorAR, blksz);
        cudaDeviceSynchronize();
        if (status != CUSPARSE_STATUS_SUCCESS) { //CUSPARSE_STATUS_INTERNAL_ERROR CUSPARSE_STATUS_SUCCESS CUSPARSE_STATUS_EXECUTION_FAILED
            printf("Matrix-matrix multiplication failed: %d\n", status);
            return 1;
        }
        const double end = get_seconds();

        time_taken += (end - start);

        transpose_GPU(d_activeBlockVectorAR, d_temp_actAR, blksz, currentBlockSize);
    //     /* print final results (z) */
        cudaStat = cudaMemcpy(activeBlockVectorAR+(i * block_width * currentBlockSize), d_temp_actAR, blksz * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess)  {
            printf("Memcpy from Device to Host failed");
            return 1;
        }
        cudaDeviceSynchronize();
    }
    
    printf("          Tiled SpMM time =  %0.3lf seconds\n", time_taken);

    double flops_computed = 1e-9 * 2.0 * nnonzero * currentBlockSize;
    
	printf("           FLOPs computed =  %0.3lf GF\n", flops_computed);
	printf("             GFLOP/s rate =  %0.3lf GF/s\n", flops_computed / time_taken);
    printf("Total memory for Matrices =  %0.3lf + %0.3lf = %0.3lf GB\n", sparse_mtx_total, dense_mtx_total, sparse_mtx_total + dense_mtx_total);
    printf("  GPU memory for Matrices =  %0.3lf + %0.3lf = %0.3lf GB\n", max_sparse_block_size, dense_mtx_size, max_sparse_block_size+dense_mtx_size);
    // double matrix_memory = 1e-9 * nnonzero * sizeof(double) + 1e-9 * nnonzero * sizeof(int) + 1e-9 * (numrows + 1) * sizeof(int) + 1e-9 * 2 * numrows * currentBlockSize * sizeof(double);
    // printf("Memory for Matrices =  %0.3lf GB\n", matrix_memory);
     
	printf("===============================================================\n");
    printf("\n%d,%d,%d,%d,%0.3lf,%0.3lf,%0.3lf,%0.3lf\n\n", numrows,numcols,currentBlockSize,block_width,matrix_memory,gpu_memory,time_taken,flops_computed / time_taken);

    // printf("Printing first 5 rows of activeBlockVectorAR\n");
    // for(i = 0 ; i < 5 ; i++){
    //     for (j = 0 ; j < currentBlockSize ; j++){
    //         printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("Printing last 5 rows of activeBlockVectorAR\n");
    // for(i = numrows - 5 ; i < numrows ; i++){
    //     for (j = 0 ; j < currentBlockSize ; j++){
    //         printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("\n--------- checking first row and first column multiplicaiton manually ------------\n\n");
    // printf("ia[0]: %d ia[1]: %d\n", ia[0], ia[1]);
    // double sum = 0;
    // for(int i = ia[0] ; i < ia[1] ; i++)
    //     sum += acsr[i] * activeBlockVectorR[ja[i] * currentBlockSize + 0];
    // printf("sum: %0.6lf\n", sum);
#endif

#if defined(USE_CPU)
    spmm_csr(numrows, numcols, currentBlockSize, ia, ja, acsr, activeBlockVectorR, activeBlockVectorAR);
    
    printf("Printing first 5 rows of activeBlockVectorAR\n");
    for(i = 0 ; i < 5 ; i++){
        for (j = 0 ; j < currentBlockSize ; j++){
            printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Printing last 5 rows of activeBlockVectorAR\n");
    for(i = numrows - 5 ; i < numrows ; i++){
        for (j = 0 ; j < currentBlockSize ; j++){
            printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#if defined(USE_OPENMP)
    spmm_csr_OpenMP(numrows, numcols, currentBlockSize, ia, ja, acsr, activeBlockVectorR, activeBlockVectorAR);
    printf("Printing first 5 rows of activeBlockVectorAR\n");
    for(i = 0 ; i < 5 ; i++){
        for (j = 0 ; j < currentBlockSize ; j++){
            printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#if defined(USE_MKL)
   char matdescra[6];
    char transA = 'n';
    matdescra[0] = 'g';
    matdescra[1] = 'l';
    matdescra[2] = 'u';
    matdescra[3] = 'c';
   
    mkl_dcsrmm(&transA, &M, &currentBlockSize, &M, &alpha, matdescra, acsr, ja, ia, ia+1, activeBlockVectorR, &currentBlockSize, &beta, activeBlockVectorAR, &currentBlockSize);
    
    printf("Printing first 5 rows of activeBlockVectorAR\n");
    for(i = 0 ; i < 5 ; i++){
        for (j = 0 ; j < currentBlockSize ; j++){
            printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Printing last 5 rows of activeBlockVectorAR\n");
    for(i = numrows - 5 ; i < numrows ; i++){
        for (j = 0 ; j < currentBlockSize ; j++){
            printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#if defined(CHECK_SANITY)
    cudaError_t cuberror;
    double *d_xrem, *d_temp3, *d_newX;
    int *d_irem, *d_colptrs;
    cusparseMatDescr_t descr = 0;

    cuberror = cudaMalloc ((void**)&d_activeBlockVectorR, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_activeBlockVectorAR, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorAR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_temp3, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_temp3\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_newX, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_newX\n"); return 0; }

    cuberror = cudaMalloc ((void**)&d_xrem, nnonzero * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_xrem\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_irem, nnonzero * sizeof(int));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_irem\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_colptrs, (numrows+1) * sizeof(int));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_colptrs\n"); return 0; }

    const double cudaAlpha = 1.0;
    const double cudaBeta = 0.0;
    const double cudaBetaOne = 1.0;
        
    cublasStatus_t cubstat;
    cusparseStatus_t status;
    cusparseHandle_t cusparseHandle = 0;
    status = cusparseCreate(&cusparseHandle);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("CUSPARSE Library initialization failed");
        return 1;
    }
    
    cuberror = cudaMemcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorR ==> %d\n", cuberror); return 0; }
    cuberror = cudaMemcpy(d_xrem, acsr, nnonzero * sizeof(double), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed xrem ==> %d\n", cuberror); return 0; }
    cuberror = cudaMemcpy(d_irem, ja, nnonzero * sizeof(int), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed irem ==> %d\n", cuberror); return 0; }
    cuberror = cudaMemcpy(d_colptrs, ia, (numrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed d_colptrs ==> %d\n", cuberror); return 0; }

    /* create and setup matrix descriptor */
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    transpose_GPU(d_activeBlockVectorR, d_newX, currentBlockSize, M);

    status = cusparseDcsrmm(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, currentBlockSize, M,
                            nnonzero, &cudaAlpha, descr, d_xrem, d_colptrs, d_irem,
                            d_newX, M, &cudaBeta, d_temp3, M);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) 
    printf("cusparseDcsrmm status: %d\n", status);

    transpose_GPU(d_temp3, d_activeBlockVectorAR, M, currentBlockSize);

    cuberror = cudaMemcpy(activeBlockVectorAR, d_activeBlockVectorAR, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuberror != cudaSuccess)  {
        printf("Memcpy from Device to Host failed");
        return 1;
    }
    cudaDeviceSynchronize();
    printf("Printing first 5 rows of activeBlockVectorAR\n");
    for(i = 0 ; i < 5 ; i++){
        for (j = 0 ; j < currentBlockSize ; j++){
            printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Printing last 5 rows of activeBlockVectorAR\n");
    for(i = numrows - 5 ; i < numrows ; i++){
        for (j = 0 ; j < currentBlockSize ; j++){
            printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#if defined(USE_UM)
    printf("===============================================================\n");
    
    cudaError_t cuberror;
    double *newX;
    // int *d_irem, *d_colptrs;
    cusparseMatDescr_t descr = 0;
    newX = (double *) malloc(M * currentBlockSize * sizeof(double));
    // cuberror = cudaMalloc ((void**)&d_activeBlockVectorR, M * currentBlockSize * sizeof(double));
    // if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorR\n"); return 0; }
    // cuberror = cudaMalloc ((void**)&d_activeBlockVectorAR, M * currentBlockSize * sizeof(double));
    // if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorAR\n"); return 0; }
    // cuberror = cudaMalloc ((void**)&d_temp3, M * currentBlockSize * sizeof(double));
    // if( cuberror != 0 ){ printf("cudaMalloc Filed d_temp3\n"); return 0; }
    // cuberror = cudaMalloc ((void**)&d_newX, M * currentBlockSize * sizeof(double));
    // if( cuberror != 0 ){ printf("cudaMalloc Filed d_newX\n"); return 0; }

    // cuberror = cudaMalloc ((void**)&d_xrem, nnonzero * sizeof(double));
    // if( cuberror != 0 ){ printf("cudaMalloc Filed d_xrem\n"); return 0; }
    // cuberror = cudaMalloc ((void**)&d_irem, nnonzero * sizeof(int));
    // if( cuberror != 0 ){ printf("cudaMalloc Filed d_irem\n"); return 0; }
    // cuberror = cudaMalloc ((void**)&d_colptrs, (numrows+1) * sizeof(int));
    // if( cuberror != 0 ){ printf("cudaMalloc Filed d_colptrs\n"); return 0; }

    const double cudaAlpha = 1.0;
    const double cudaBeta = 0.0;
    const double cudaBetaOne = 1.0;
        
    cublasStatus_t cubstat;
    cusparseStatus_t status;
    cusparseHandle_t cusparseHandle = 0;
    status = cusparseCreate(&cusparseHandle);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("CUSPARSE Library initialization failed");
        return 1;
    }
    
    // cuberror = cudaMemcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
    // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorR ==> %d\n", cuberror); return 0; }
    // cuberror = cudaMemcpy(d_xrem, acsr, nnonzero * sizeof(double), cudaMemcpyHostToDevice);
    // if( cuberror != 0 ){ printf("cudaMemcpy failed xrem ==> %d\n", cuberror); return 0; }
    // cuberror = cudaMemcpy(d_irem, ja, nnonzero * sizeof(int), cudaMemcpyHostToDevice);
    // if( cuberror != 0 ){ printf("cudaMemcpy failed irem ==> %d\n", cuberror); return 0; }
    // cuberror = cudaMemcpy(d_colptrs, ia, (numrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    // if( cuberror != 0 ){ printf("cudaMemcpy failed d_colptrs ==> %d\n", cuberror); return 0; }

    /* create and setup matrix descriptor */
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    transpose_GPU(activeBlockVectorR, temp_actR, currentBlockSize, M);
    const double start = get_seconds();
    status = cusparseDcsrmm(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, currentBlockSize, M,
                            nnonzero, &cudaAlpha, descr, acsr, ia, ja,
                            temp_actR, M, &cudaBeta, newX, M);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) 
        printf("cusparseDcsrmm status: %d\n", status);
    const double end = get_seconds();

    transpose_GPU(newX, activeBlockVectorAR, M, currentBlockSize);

    const double time_taken = (end - start);
    
	printf("        Unified SpMM time =  %0.3lf seconds\n", time_taken);

    // double matrix_memory = 1e-9 * nnonzero * sizeof(double) + 1e-9 * nnonzero * sizeof(int) + 1e-9 * (numrows + 1) * sizeof(int) + 1e-9 * 2 * numrows * currentBlockSize * sizeof(double);
    double flops_computed = 1e-9 * 2.0 * nnonzero * currentBlockSize;
    
	printf("           FLOPs computed =  %0.3lf GF\n", flops_computed);
	printf("             GFLOP/s rate =  %0.3lf GF/s\n", flops_computed / time_taken);
    printf("Memory for Matrices =  %0.3lf GB\n", matrix_memory);
    
	printf("===============================================================\n");
    printf("\n%d,%d,%d,%0.3lf,%0.3lf,%0.3lf\n\n", numrows, numcols, currentBlockSize, matrix_memory, time_taken, flops_computed / time_taken);
    // cuberror = cudaMemcpy(activeBlockVectorAR, d_activeBlockVectorAR, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
    // if (cuberror != cudaSuccess)  {
    //     printf("Memcpy from Device to Host failed");
    //     return 1;
    // }
    
    // cudaDeviceSynchronize();
    
    // printf("Printing first 5 rows of activeBlockVectorAR\n");
    // for(i = 0 ; i < 5 ; i++){
    //     for (j = 0 ; j < currentBlockSize ; j++){
    //         printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("Printing last 5 rows of activeBlockVectorAR\n");
    // for(i = numrows - 5 ; i < numrows ; i++){
    //     for (j = 0 ; j < currentBlockSize ; j++){
    //         printf("%.6lf ", activeBlockVectorAR[i * currentBlockSize + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    
#endif


    return 0;
}


//https://stackoverflow.com/questions/51345922/number-of-operations-for-sparsedense-matrix-multiplication
