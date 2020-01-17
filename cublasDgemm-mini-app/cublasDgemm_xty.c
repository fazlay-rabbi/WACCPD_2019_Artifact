
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

double get_seconds() {
	struct timeval now;
	gettimeofday(&now, NULL);

	const double seconds = (double) now.tv_sec;
	const double usec    = (double) now.tv_usec;

	return seconds + (usec * 1.0e-6);
}

int main(int argc, char* argv[]) 
{
	int N = atoi(argv[1]);
	int repeats = 1;
	double alpha = 1.0;
	double beta  = 1.0;
	int blksz;

	int blocksize = atoi(argv[2]);
	int block_width = atoi(argv[3]);
	
	printf("                  N =  %d\n", N);
    printf("              Alpha =  %0.3lf\n", alpha);
    printf("              Beta  =  %0.3lf\n", beta);
	printf("            Repeat  =  %d\n", repeats);
	printf("         blocksize  =  %d\n", blocksize);
	printf("         Tile size  =  %d\n", block_width);
	

	printf("Allocating Matrices...\n");
	double* DGEMM_RESTRICT matrixA = (double*) malloc(sizeof(double) * N * blocksize);
	double* DGEMM_RESTRICT matrixB = (double*) malloc(sizeof(double) * N * blocksize);
	double* DGEMM_RESTRICT matrixC = (double*) malloc(sizeof(double) * blocksize * blocksize);

	printf("Allocation complete, populating with values...\n");

	int i, j, k, r;
	double gpu_memory;
	
	#pragma omp parallel for
	for(i = 0 ; i < N ; i++) 
	{
		for(j = 0 ; j < blocksize ; j++) 
		{
			matrixA[i * blocksize + j] = 0.25;
			matrixB[i * blocksize + j] = 0.25;
		}
	}

	printf("Performing multiplication...\n");

	printf("\n");
	printf("===============================================================\n");

	const double start = get_seconds();
	double sum = 0;
	
#if defined( USE_CUBLAS ) 
  	double *devPtrA, *devPtrB, *devPtrC;
  	cudaMalloc ((void**)&devPtrA, block_width * blocksize * sizeof(double));
  	cudaMalloc ((void**)&devPtrB, block_width * blocksize * sizeof(double));
  	cudaMalloc ((void**)&devPtrC, blocksize * blocksize * sizeof(double));

  	cublasStatus_t cubstat;
  	cublasHandle_t handle;
	cudaError_t cuberror;
  	cubstat = cublasCreate(&handle);
  	if( cubstat != CUBLAS_STATUS_SUCCESS ){ printf("HandleCreationFailure\n"); return 0; }	
#endif

	// Repeat multiple times
	for(r = 0; r < repeats; r++) 
	{

	#if defined( USE_MKL ) || defined (USE_CBLAS)
		// I didn't finish the Intel MKL version on CPU
		beta = 0.0;
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				blocksize, blocksize, N, alpha, matrixA, blocksize, matrixB, blocksize, beta, matrixC, blocksize);
	
	#elif defined( USE_NVBLAS )
		char transA = 'N';
		char transB = 'N';

		dgemm(&transA, &transB, &N, &N, &N, &alpha, matrixA, &N,
			matrixB, &N, &beta, matrixC, &N);

	#elif defined(USE_CUBLAS)
		// This is my tiling version
		int nrowblk = ceil(1.0 * N / block_width);
		printf("            nrowblk =  %d\n", nrowblk);
		gpu_memory = 1.0 * 1e-9 * ((2 * blocksize * block_width * sizeof(double) + blocksize * blocksize * sizeof(double)));
		printf("    GPU memory used =  %0.3lf GB\n", gpu_memory);

		cudaMemset(devPtrC, 0.0, blocksize * blocksize * sizeof(double));

		for(i = 0 ; i < nrowblk ; i++)
		{
			blksz = block_width;
			if(i * block_width + blksz > N)
				blksz = N - i * block_width;
			
			cuberror = cudaMemcpy((void *)devPtrA, matrixA + (i * block_width * blocksize), blksz * blocksize * sizeof(double), cudaMemcpyHostToDevice);
			if(cuberror != 0) { printf("cudaMalloc Filed to copy devPtrA, block : %d\n", i); return 0; }

			cuberror = cudaMemcpy((void *)devPtrB, matrixB + (i * block_width * blocksize), blksz * blocksize * sizeof(double), cudaMemcpyHostToDevice);
			if(cuberror != 0) { printf("cudaMalloc Filed to copy devPtrB, block : %d\n", i); return 0; }
			
			cudaDeviceSynchronize();

			cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, blocksize, blocksize, blksz, 
					&alpha, devPtrB, blocksize, devPtrA, blocksize, &beta, devPtrC, blocksize); 
			if(cubstat != CUBLAS_STATUS_SUCCESS)
				printf("cublasDgemm status: %d\n",cubstat);
			cudaDeviceSynchronize();
		}	
		cuberror = cudaMemcpy(matrixC, devPtrC, blocksize * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
    	if( cuberror != 0 ){ printf("cudaMemcpy failed temp at blok_id: %d errocode: %d\n", i, cuberror);}
	#elif defined( USE_UM )
		cublasStatus_t cubstat;
		cublasHandle_t handle;
		beta = 0.0;
		cubstat = cublasCreate(&handle);
		if( cubstat != CUBLAS_STATUS_SUCCESS ){ printf("HandleCreationFailure\n"); return 0; }
		cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, blocksize, blocksize, N, 
					&alpha, matrixB, blocksize, matrixA, blocksize, &beta, matrixC, blocksize); 
		if(cubstat != CUBLAS_STATUS_SUCCESS)
			printf("cublasDgemm status: %d\n",cubstat);
	#else
	
		// #pragma omp parallel for private(sum)
		#pragma acc parallel loop
		for(i = 0; i < N; i++) {
			#pragma acc loop
			for(j = 0; j < N; j++) {
				sum = 0;
				for(k = 0; k < N; k++) {
					sum += matrixA[i * N + k] * matrixB[k * N + j];
				}
				matrixC[i * N + j] = (alpha * sum) + (beta * matrixC[i * N + j]);
			}
		}
	#endif
	
	} // end repeat loop

#if defined( USE_CUBLAS ) || defined(USE_CUBLAS_TILE)
	cudaFree (devPtrA);
	cudaFree (devPtrB);
	cudaFree (devPtrC);
	cublasDestroy(handle); 
#elif defined( USE_UM )
  	cudaDeviceSynchronize();
#endif
	const double end = get_seconds();

	double matrix_memory = 1e-9 * 2 * N * blocksize * sizeof(double) + 1e-9 * blocksize * blocksize * sizeof(double);
	const double time_taken = (end - start);
	const double flops_computed = 1e-9 * N * blocksize * blocksize * 2.0 * (double)(repeats) + 1e-9 * blocksize * blocksize * 2 * (double)(repeats);
	printf("      Multiply time =  %0.3lf seconds\n", time_taken);
	printf("     FLOPs computed =  %0.3lf GF\n", flops_computed);
	printf("       GFLOP/s rate =  %0.3lf GF/s\n", flops_computed / time_taken);
	printf("Memory for Matrices =  %0.3lf GB\n", matrix_memory);

	printf("===============================================================\n");
	printf("\n");

	// printf("Printing Result matrix: \n");
	// for(i = 0 ; i < blocksize ; i++)
	// {
	// 	for(j = 0 ; j < blocksize ; j++)
	// 		printf("%0.1lf ", matrixC[i * blocksize + j]);
	// 	printf("\n");
	// }
	const double allowed_margin = 1.0e-8;
	for(i = 0 ; i < blocksize ; i++)
	{
		for(j = 0 ; j < blocksize ; j++)
		{
			if((matrixC[i * blocksize + j] - N * 0.25 *0.25) > allowed_margin)
			{
				printf(" -> Solution check FAILED.\n");
				return 0;
			}
		}
	}
	printf(" -> Solution check PASSED successfully.\n\n");
		
#if defined(USE_CUBLAS)
	printf("%d,%d,%d,%0.3lf,%0.3lf,%0.3lf,%0.3lf\n\n", N, blocksize, block_width, matrix_memory, gpu_memory, time_taken, flops_computed / time_taken);
#elif defined(USE_UM)
	printf("%d,%d,%0.3lf,%0.3lf,%0.3lf\n\n", N, blocksize, matrix_memory, time_taken, flops_computed / time_taken);
#endif

	free(matrixA);
	free(matrixB);
	free(matrixC);

	return 0;
}
