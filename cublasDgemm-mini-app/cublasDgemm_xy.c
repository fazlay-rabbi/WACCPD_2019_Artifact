#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifdef USE_CBLAS
#include "cblas.h"
#elif USE_NVBLAS
#include "nvblas.h"
#elif USE_MKL
// #include "mkl.h"
#include "cblas.h"
#elif USE_CUBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
#elif USE_UM
#include <cuda_runtime.h>
#include "cublas_v2.h"
#elif USE_CUBLAS_TILE
#include <cuda_runtime.h>
#include "cublas_v2.h"

#endif

#define DGEMM_RESTRICT __restrict__

double get_seconds() 
{
	struct timeval now;
	gettimeofday(&now, NULL);

	const double seconds = (double) now.tv_sec;
	const double usec    = (double) now.tv_usec;

	return seconds + (usec * 1.0e-6);
}

int main(int argc, char* argv[]) 
{
	int N = 30, M, P;
	int repeats = 1;

	double alpha = 1.0;
	double beta  = 0.0;
	double gpu_memory;

	int blksz;

	N = atoi(argv[1]);
	M = atoi(argv[2]);
	P = atoi(argv[3]);
		

	int block_width = atoi(argv[4]);
  	
	printf("                  N =  %d\n", N);
	printf("                  M =  %d\n", M);
	printf("                  P =  %d\n", P);
    printf("              Alpha =  %0.3lf\n", alpha);
    printf("              Beta  =  %0.3lf\n", beta);
	printf("            Repeat  =  %d\n", repeats);
#if defined(USE_CUBLAS)
	printf("         Tile size  =  %d\n", block_width);
#endif
	double* DGEMM_RESTRICT matrixA = (double*) malloc(sizeof(double) * N * M);
	double* DGEMM_RESTRICT matrixB = (double*) malloc(sizeof(double) * M * P);
	double* DGEMM_RESTRICT matrixC = (double*) malloc(sizeof(double) * N * P);


	double* DGEMM_RESTRICT temp = (double*) malloc(sizeof(double) * block_width * M);

	printf("Allocation complete, populating with values...\n");

	int i, j, k, r;

	#pragma omp parallel for
	for(i = 0; i < N; i++) {
		for(j = 0; j < M; j++) {
			matrixA[i * M + j] = 2.0;
		}
	}
	
	#pragma omp parallel for
	for(i = 0; i < M; i++) {
		for(j = 0; j < P; j++) {
			matrixB[i * P + j] = 0.5;
		}
	}
	#pragma omp parallel for
	for(i = 0; i < N; i++) {
		for(j = 0; j < P; j++) {
			matrixC[i * P + j] = 0.0;
		}
	}

	const double start = get_seconds();

#if defined( USE_CUBLAS ) 
	double *devPtrA, *devPtrB, *devPtrC;
	cudaMalloc ((void**)&devPtrA, block_width * M * sizeof(double));
	cudaMalloc ((void**)&devPtrB, M * P * sizeof(double));
	cudaMalloc ((void**)&devPtrC, block_width * P * sizeof(double));

	cublasStatus_t cubstat;
	cublasHandle_t handle;
	cudaError_t cuberror;
	cubstat = cublasCreate(&handle);
	if( cubstat != CUBLAS_STATUS_SUCCESS ){ printf("HandleCreationFailure - 1 \n"); return 0; }

	cubstat |= cudaMemcpy((void *)devPtrB, matrixB, M * P * sizeof(double), cudaMemcpyHostToDevice);
	if( cubstat != CUBLAS_STATUS_SUCCESS ){ printf("cudaMemcpyFailure\n"); return 0; }
#endif

	// Repeat multiple times
	for(r = 0 ; r < repeats; r++) 
	{

#if defined(USE_MKL) || defined (USE_CBLAS)
	// I didn't implement the MKL verison
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, alpha, matrixA, N, matrixB, N, beta, matrixC, N);
#elif defined( USE_NVBLAS )
		char transA = 'N';
		char transB = 'N';

		dgemm(&transA, &transB, &N, &N, &N, &alpha, matrixA, &N,
			matrixB, &N, &beta, matrixC, &N);
#elif defined(USE_CUBLAS)
	printf("===============================================================\n");
	
	int nrowblk = ceil(1.0 * N/block_width);
	printf("            nrowblk =  %d\n", nrowblk);
	gpu_memory = 1e-9 * ((block_width * M * sizeof(double) + block_width * P * sizeof(double) + M * P * sizeof(double)));
	printf("    GPU memory used =  %0.3lf GB\n", gpu_memory);
	for(i = 0 ; i < nrowblk ; i++)
	{
		blksz = block_width;
		if(i * block_width + blksz > N)
			blksz = N - i * block_width;
		cuberror = cudaMemcpy((void *)devPtrA, matrixA+(i*block_width*M), blksz * M * sizeof(double), cudaMemcpyHostToDevice);
		if(cuberror != 0){ printf("cudaMalloc Filed to copy matrixA block : %d errorcode: %d\n", i, cuberror); return 0; }
		cudaDeviceSynchronize();
		
		cubstat = cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, P, blksz, M,
	              &alpha, devPtrB, P, devPtrA, M, &beta, devPtrC, P);
		cudaDeviceSynchronize();
		if(cubstat != CUBLAS_STATUS_SUCCESS){ printf("cublasDgemm Failed in Tiling\n"); return 0; }

		cuberror = cudaMemcpy(matrixC+(i*block_width*P), devPtrC, blksz * P * sizeof(double), cudaMemcpyDeviceToHost);
    	if( cuberror != 0 ){ printf("cudaMemcpy failed devPtrC at blok_id: %d errocode: %d\n", i, cuberror);}
		cudaDeviceSynchronize();
	}

#elif defined(USE_UM)
	printf("===============================================================\n");
	cublasStatus_t cubstat;
	cublasHandle_t handle;
	cubstat = cublasCreate(&handle);
	if(cubstat != CUBLAS_STATUS_SUCCESS){ printf("HandleCreationFailure - 2\n"); return 0; }
	
	// parameter meaning and appropriate position
	// C(m,n) = A(m,k) * B(k,n)
	// int lda=m,ldb=k,ldc=m;
	// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	
	cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, P, M,
	              &alpha, matrixA, N, matrixB, M, &beta, matrixC, N);
	cudaDeviceSynchronize();
	if(cubstat != CUBLAS_STATUS_SUCCESS){ printf("cublasDgemm Failed in UM\n"); return 0; }
#else
	// I didn't implement the loop parallel version
	// #pragma omp parallel for private(sum)
	#pragma acc parallel loop
	for(i = 0; i < N; i++) {
		#pragma acc loop
		for(j = 0; j < N; j++) {
			sum = 0;

			for(k = 0; k < N; k++) {
				sum += matrixA[i*N + k] * matrixB[k*N + j];
			}
			matrixC[i*N + j] = (alpha * sum) + (beta * matrixC[i*N + j]);
		}
	}
#endif

	} // end of repeat loop

#if defined( USE_CUBLAS ) || defined(USE_CUBLAS_TILE)
	cudaFree (devPtrA);
	cudaFree (devPtrB);
	cudaFree (devPtrC);
	cublasDestroy(handle); 
#endif
	const double end = get_seconds();

	double matrix_memory = 1e-9 * N * M * sizeof(double) + 1e-9 * M * P * sizeof(double) + 1e-9 * N * P * sizeof(double);
	const double time_taken = (end - start);
	const double flops_computed = 1e-9 * 2 * N * M * P * (double)(repeats) + 1e-9 * 2 * N * P * (double)(repeats);
	
	printf("      Multiply time =  %0.3lf seconds\n", time_taken);
	printf("     FLOPs computed =  %0.3lf GF\n", flops_computed);
	printf("       GFLOP/s rate =  %0.3lf GF/s\n", flops_computed / time_taken);
	printf("Memory for Matrices =  %0.3lf GB\n", matrix_memory);

	printf("===============================================================\n");
	printf("\n");

	const double allowed_margin = 1.0e-8;
	for(i = 0 ; i < N ; i++)
	{
		for(j = 0 ; j < P ; j++)
		{
			if((matrixC[i * P + j] - M * 2 * 0.50) > allowed_margin)
			{
				printf(" -> Solution check FAILED.\n");
				return 0;
			}
		}
	}
	printf(" -> Solution check PASSED successfully.\n\n");
#if defined(USE_CUBLAS)
	printf("%d,%d,%d,%d,%0.3lf,%0.3lf,%0.3lf,%0.3lf\n\n", N, M, P, block_width, matrix_memory, gpu_memory, time_taken, flops_computed / time_taken);
#elif defined(USE_UM)
	printf("%d,%d,%d,%0.3lf,%0.3lf,%0.3lf\n\n", N, M, P, block_width, matrix_memory, time_taken, flops_computed / time_taken);
#endif

	free(matrixA);
	free(matrixB);
	free(matrixC);

	return 0;
}
