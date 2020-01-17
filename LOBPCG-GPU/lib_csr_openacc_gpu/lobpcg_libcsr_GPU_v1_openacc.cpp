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

#ifdef USE_CUBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif
#define DGEMM_RESTRICT __restrict__

// #include <mkl.h>
#include <omp.h>
#include "lib_csr.h"

#include <cblas.h>
#include <lapacke.h>

double get_seconds() 
{
	struct timeval now;
	gettimeofday(&now, NULL);

	const double seconds = (double) now.tv_sec;
	const double usec    = (double) now.tv_usec;

	return seconds + (usec * 1.0e-6);
}


int main(int argc, char *argv[])
{
    int M, N, index = 0, blocksize;
    int block_width;
    double *A, *blockVectorX;
    double *at, *bt;
    double residualTolerance = 0.0001;
    long maxIterations = 10;
    int constraintStyle = 0; //operatorB not used
    long iterationNumber;

    int info;

    //cout settings
    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);

    stringstream s(argv[1]);
    s >> blocksize;
    // stringstream s1(argv[2]);
    // s1 >> block_width;
    

    int currentBlockSize = blocksize, prevCurrentBlockSize = blocksize;

    

    //----- lib spmm params 
    char matdescra[6];
    char transA = 'n';
    matdescra[0] = 'g';
    matdescra[1] = 'l';
    matdescra[2] = 'u';
    matdescra[3] = 'c';
    
    double alpha = 1.0, beta = 0.0;

    //---- dportf_ & dsygv_ params
    const char jobvl = 'N';
    const char jobvr = 'V';
    const char jobz = 'V';
    const char uplo = 'U';
    const char uplo_dlacpy = ' ';
    const int itype = 1;
    double work_query;
    int lwork;
    double *work;

    int i,j,k;

    double *xrem;
    double *acsr;
    block<double> *matrixBlock;
    char *filename = argv[2] ; 
    wblk = block_width; 
    // read_custom(filename,xrem);
    //read_custom_csr(filename, acsr);
    // printf("Finish Reading CUS file\n");

    // *--------- Reading CSR from binary file ---------* //
    ifstream file (filename, ios::in|ios::binary);
    if (file.is_open())
    {
        file.read ((char*)&numrows,sizeof(numrows));
        cout << "row: "<<numrows<<endl;
        file.read(reinterpret_cast<char*>(&numcols), sizeof(numcols));
        cout << "colum: " << numcols << endl;

        file.read(reinterpret_cast<char*>(&nnonzero), sizeof(nnonzero));
        cout << "non zero: " << nnonzero << endl;

        ia = (int *) malloc((numrows + 1) * sizeof(int)); //colsptr
        ja = (int *) malloc(nnonzero * sizeof(int)); //irem
        acsr = (double *) malloc(nnonzero * sizeof(double)); //xrem

        // file.read(reinterpret_cast<char*>(ia), (numrows + 1) * sizeof(int));
        // cout << "finished reading ia"<<endl;
        // file.read(reinterpret_cast<char*>(ja), nnonzero * sizeof(int));
        // cout << "finished reading ja"<<endl;
        // file.read(reinterpret_cast<char*>(acsr), nnonzero * sizeof(double));
        // cout << "finished reading acsr"<<endl;

        i = 0;
        while(!file.eof() && i <= numrows)
        {
            file.read(reinterpret_cast<char*>(&j), sizeof(j)); //irem(j)
            ia[i++] = j;
        }
        cout << "finished reading ia"<<endl;
        i = 0;
        while(!file.eof() && i < nnonzero)
        {
            file.read(reinterpret_cast<char*>(&j), sizeof(j)); //irem(j)
            ja[i++] = j;
        }
        cout << "finished reading ja"<<endl;
        i = 0;
        double d;
        while(!file.eof() && i < nnonzero)
        {
            file.read(reinterpret_cast<char*>(&d), sizeof(d)); //irem(j)
            acsr[i++] = d;
        }  
        cout << "finished reading acsr"<<endl;
    }
    file.close();
    cout << "# of vector block: " << blocksize  << endl;
    
    cout << "ia[0]: " << ia[0] << " ia[last]: " << ia[numrows] << endl;
    cout << "ja[0]: " << ja[0] << " ja[last]: " << ja[nnonzero-1] << endl;
    cout << "acsr[0]: " << acsr[0] << " acsr[last]: " << acsr[nnonzero-1] << endl;
    printf("numrows: %d numcols: %d nnonzero: %d\n", numrows, numcols, nnonzero);
    
    #pragma omp parallel
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }

    //-- deleting CSC storage memory ------
    //delete []colptrs;
    //delete []irem;
    //delete []xrem;

    M = numrows;
    N = numcols;

    //timing variables
    int numTasks = 19;
    vector<string> function_name{"LOOPS", "X*Y", "Xt*Y", "ADD", "SUB", "MULT", "SPMM", "GET", "UPDATE", "dsygv", "DLACPY", "INVERSE", "TRANSPOSE", "mat_copy", "dpotrf", "memset", "SUMSQRT", "diag", "cudaMemcpy"};
    double **taskTiming = (double **) malloc(sizeof(double *) * maxIterations);
    for(i = 0 ; i < maxIterations ; i++)
        taskTiming[i] = (double *) malloc(sizeof(double) * numTasks);
    
    #pragma omp parallel for default(shared) private(j)
    for(i = 0 ; i < maxIterations ; i++)
    {
        for(j = 0 ; j < numTasks ; j++)
        {
            taskTiming[i][j] = 0.0;
        }
    }

    double tstart, tend, temp1Time;
    double loop_start_time = 0, loop_finish_time = 0;
    double iteraton_time = 0, iteraton_start_time = 0;

    blockVectorX = (double *) malloc(M * blocksize * sizeof(double));

    int job_dcsrcsc[] = {1, 0, 0, 0, 0, 1}; 
    int dcsrcsc_info = -1;
    
    // acsr = (double *) malloc(nnonzero * sizeof(double)); //new double[nnonzero](); //xrem
    // ja = (int *) malloc(nnonzero * sizeof(int)); //new int[nnonzero](); //irem
    // ia = (int *) malloc((numrows + 1) * sizeof(int)); //new int[numrows + 1](); //colsptr

    // tstart = omp_get_wtime();
    
    // mkl_dcsrcsc(job_dcsrcsc, &numrows, acsr, ja, ia, xrem, irem, colptrs, &dcsrcsc_info);
    
    //printf("mkl_dcsrcsc: %lf sec.\n", omp_get_wtime() - tstart);

    //-- deleting CSC storage memory ------
    // delete []colptrs;
    // delete []irem;
    // delete []xrem;

    //std::fstream blockVectorXfile("MatX100.txt", std::ios_base::in);
    srand(0);
   //#pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < blocksize ; j++)
        {
            //blockVectorXfile>>blockVectorX[i*blocksize+j];
            blockVectorX[i * blocksize + j] = (double)rand()/(double)RAND_MAX;
            //blockVectorX[i*blocksize+j]= -1.00 + rand() % 2 ;
        }
    }

    //cout<<"finished reading X"<<endl;

    //******** memory allocation for matrices ********

    double *blockVectorAX = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorR = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorAR = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorP = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorAP = (double *) malloc(M * blocksize * sizeof(double));
    

    double *activeBlockVectorR = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorAR = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorAP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *temp3 = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *newX = (double *) malloc(M * blocksize * sizeof(double));
    
    int gramASize = blocksize + currentBlockSize + currentBlockSize;
    double *gramA = (double *) malloc(gramASize * gramASize * sizeof(double));
    double *gramB = (double *) malloc(gramASize * gramASize * sizeof(double));
    double *eigen_value;

    //--- modified new

    
    //double *newAX = new double[M*blocksize]();
    //double *temp1 = new double[M*blocksize]();

    //double *newActP = new double[M*currentBlockSize]();
    //double *tempMultResult=new double[M*currentBlockSize]();
    
    double *residualNorms = (double *) malloc(blocksize * sizeof(double));
    double *gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *trans_gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *temp2 = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *trans_gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *gramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *transGramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *transGramRAP= (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramRAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramPAP= (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *identity_PAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *transGramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *identity_BB = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *gramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));

    double *zeros_B_CB = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *zeros_CB_B = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    
    std::memset(zeros_B_CB, 0.0, blocksize * currentBlockSize * sizeof(double));
    std::memset(zeros_CB_B, 0.0, blocksize * currentBlockSize * sizeof(double));

    // saveLamda[blocksize * maxIterations]
    double **saveLamda = (double **) malloc(blocksize * maxIterations * sizeof(double *));
    for(i = 0 ; i < blocksize ; i++)
        saveLamda[i] = (double *) malloc(maxIterations * sizeof(double));
    
    for(i = 0 ; i < blocksize ; i++)
        for(j = 0 ; j < maxIterations ; j++)
            saveLamda[i][j] = 0.0;

    double *loopTime = (double *) malloc(maxIterations * sizeof(double));
    for(i = 0 ; i < maxIterations ; i++)
        loopTime[i] = 0;

    //cout<<"Allocation 8"<<endl;

    //cout<<"Total allocation time: "<<omp_get_wtime()-allocation_time<<" sec."<<endl;

    //---- if 9 ----
    //gramXBX=blockVectorX'*blockVectorX;
    //[gramXBX,cholFlag]=chol(gramXBX);
    //blockVectorX = blockVectorX/gramXBX;
    
    double *gramXBX = new double[blocksize * blocksize]();

    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,blocksize,blocksize,M,1.0,blockVectorX,blocksize,blockVectorX,blocksize,0.0,gramXBX,blocksize);
    //_XTY(blockVectorX, blockVectorX, gramXBX, M, blocksize, blocksize, block_width);

    int cholFlag;
    //making the lower part of gramXBX zero

    //-- changing LAPACKE_dpotrf to dpotrf_
    // double *trans_gramXBX = new double[blocksize * blocksize]();
    
    // transpose(gramXBX, trans_gramXBX, blocksize, blocksize);
    // dpotrf_( &uplo, &blocksize, trans_gramXBX, &blocksize, &info );
    // if(info != 0 )
    // {
    //     cout<<"dpotrf: chol error!"<<endl;
    //     //exit(1);
    // }
    
    // transpose(trans_gramXBX, gramXBX, blocksize, blocksize);
    // delete []trans_gramXBX;
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR , 'U' , blocksize , gramXBX , blocksize);
    if(info != 0)
        cout << "dpotrf: chol error!" << endl;
     
    #pragma omp parallel for  private(j) default(shared)
    for(i = 0 ; i < blocksize ; i++)
    {
        for(j = 0 ; j < i ; j++)
        {
            gramXBX[i * blocksize + j] = 0.0;
        }
    }
    
    double *tempGramXBX = new double[blocksize * blocksize]();
    //int copyGramXBX = LAPACKE_dlacpy(LAPACK_ROW_MAJOR,' ',blocksize,blocksize,gramXBX,blocksize,tempGramXBX,blocksize);
    custom_dlacpy(gramXBX, tempGramXBX, blocksize, blocksize);
    
    inverse(tempGramXBX, blocksize, blocksize);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, M, blocksize, blocksize,1.0,blockVectorX, blocksize,tempGramXBX,blocksize,0.0, newX,blocksize);
    
    //copyResult=LAPACKE_dlacpy(LAPACK_ROW_MAJOR,' ',M,blocksize,tempResult,blocksize,blockVectorX,blocksize);
    custom_dlacpy(newX, blockVectorX, M, blocksize);
    delete []tempGramXBX;
    
    // if 17 
    // blockVectorAX = operatorA*blockVectorX;
    //std::memset(blockVectorAX, 0.0, sizeof(blockVectorAX));
    //tstart = omp_get_wtime();
    #pragma omp parallel for private(j) default(shared)
    for(i = 0; i < M ; i++)
    {
        for(j = 0 ; j < blocksize ; j++)
        {
            blockVectorAX[i * blocksize + j] = 0.0;
        }
    }
    //taskTiming[0] += (omp_get_wtime() - tstart); //SETZERO : 0
    
    //mkl_dcscmm(&transA, &M, &blocksize, &N, &alpha, matdescra, xrem, irem, colptrs, colptrs+1, blockVectorX, &blocksize,  &beta, blockVectorAX, &blocksize);
    //mkl_dcsrmm(&transA, &M, &blocksize, &N, &alpha, matdescra, acsr, ja, ia, ia+1, blockVectorX, &blocksize, &beta, blockVectorAX, &blocksize);
    //spmm_blkcoord(numrows, numcols, blocksize, nthrds, blockVectorX, blockVectorAX, matrixBlock);
    spmm_csr(numrows, numcols, blocksize, ia, ja, acsr, blockVectorX, blockVectorAX);
    
    //gramXAX = full(blockVectorX'*blockVectorAX);
    double *gramXAX=new double[blocksize*blocksize]();
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,blocksize,blocksize,M,1.0,blockVectorX,blocksize,blockVectorAX,blocksize,0.0,gramXAX,blocksize);
    //_XTY(blockVectorX, blockVectorAX, gramXAX, M, blocksize, blocksize, block_width);
    
    //gramXAX = (gramXAX + gramXAX')*0.5;
    double *transGramXAX=new double[blocksize*blocksize]();
    transpose(gramXAX,transGramXAX, blocksize, blocksize);
    
    make_identity_mat(identity_BB,blocksize, blocksize);
    make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize); //--> used in loop
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blocksize, blocksize, blocksize, 0.5, transGramXAX, blocksize, identity_BB, blocksize, 0.5, gramXAX, blocksize);
    
    free(transGramXAX);

    // double *temp_GramXAX = new double[blocksize*blocksize]();
    // transpose(gramXAX, temp_GramXAX, blocksize, blocksize);
    
    // //dummy call: work size query
    // lwork = -1;
    // double *tempLambda = new double[blocksize]();
    
    // dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, &work_query, &lwork, &info);
    // if(info != 0)
    // {
    //   cout<<"Error in dummy call"<<endl;
    //   //exit(1);
    // }

    // lwork = (int) work_query;
    // work = new double[lwork]();
    
    // dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, work, &lwork, &info);
    
    // if(info != 0)
    // {
    //     printf( "The algorithm failed to compute eigenvalues.\n" );
    // }
    // transpose(temp_GramXAX, gramXAX, blocksize, blocksize);
    
    // free(temp_GramXAX);
    // free(work);
    
    //[coordX,gramXAX]=eig(gramXAX,eye(blockSize));
    //lambda=diag(gramXAX);

    double *tempLambda = new double[blocksize]();
    info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, blocksize, gramXAX, blocksize, identity_BB, blocksize, tempLambda);
    if(info != 0)
        cout << "Error in dummy call" << endl;

    double *lambda = (double *) malloc(blocksize * blocksize * sizeof(double));
    diag(tempLambda, lambda, blocksize);
    
    free(tempLambda);
    
    //note: after applying dsyevd_ function gramXAX will be coordX
    //blockVectorX  =  blockVectorX*coordX;  //after this, dimension of blockVectorX will be M*blocksize
    //blockVectorAX = blockVectorAX*coordX; //blockVectorAX will remain M*blocksize
    //double *newBlockVectorX=new double[M*blocksize]();
    
    double *coordX;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,blocksize,blocksize,1.0,blockVectorX,blocksize,gramXAX,blocksize,0.0,newX,blocksize);
    
    //copyResult=LAPACKE_dlacpy(LAPACK_ROW_MAJOR,' ',M,blocksize,newBlockVectorX,blocksize,blockVectorX,blocksize); //blockVectorX=newBlockVectorX
    custom_dlacpy(newX, blockVectorX, M, blocksize);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.00, blockVectorAX, blocksize, gramXAX, blocksize, 0.00, newX, blocksize);
    custom_dlacpy(newX, blockVectorAX, M, blocksize);
    free(gramXAX);
    

    
    int *activeMask = (int *) malloc(blocksize * sizeof(int));

    #pragma omp parallel for
    for(i = 0 ; i < blocksize ; i++)
        activeMask[i] = 1;

    iteraton_start_time = omp_get_wtime();

    int activeRSize = 0, activePSize = 0, explicitGramFlag = 0, restart = 0;


    //======== creadting device pointers for GPU start ==========//
#if defined( USE_CUBLAS )
    double *d_blockVectorP, *d_blockVectorR, *d_blockVectorX, *d_blockVectorAX; 
    double *d_blockVectorAP, *d_lambda;
    double *d_activeBlockVectorR, *d_activeBlockVectorAR, *d_temp2, *d_temp3;
    double *d_activeBlockVectorP, *d_activeBlockVectorAP;
    double *d_gramRBR, *d_gramPBP, *d_gramXAR, *d_gramRAR;
    double *d_identity_PAP, *d_transGramRAR, *d_gramXAP;
    double *d_gramPAP, *d_gramRAP, *d_gramXBP, *d_gramRBP;
    double *d_coordX, *d_newX;
    double *d_xrem, *d_residualNorms;
    int *d_irem, *d_colptrs, *d_activeMask;
    double *d_gramA, *d_gramB, *d_transGramXAR, *d_transGramXAP, *d_transGramRAP;
    double *d_identity_BB, *d_zeros_B_CB, *d_transGramXBP, *d_transGramRBP;

    const double cudaAlpha = 1.0;
    const double cudaBeta = 0.0;
    const double cudaBetaOne = 1.0;
        
    cublasStatus_t cubstat;
    cusparseStatus_t status;
    cublasHandle_t handle;
    cudaError_t cuberror;
    cusparseMatDescr_t descr = 0;
    cusparseHandle_t cusparseHandle = 0;

    cuberror = cudaMalloc ((void**)&d_blockVectorX, M * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_blockVectorX\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_blockVectorAX, M * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_blockVectorAX\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_blockVectorR, M * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_blockVectorR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_blockVectorP, M * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_blockVectorP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_blockVectorAP, M * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_blockVectorAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_newX, M * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_newX\n"); return 0; }
    

    cuberror = cudaMalloc ((void**)&d_activeBlockVectorR, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_activeBlockVectorAR, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorAR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_activeBlockVectorP, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_activeBlockVectorAP, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeBlockVectorAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_temp3, M * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_temp3\n"); return 0; }

    cuberror = cudaMalloc ((void**)&d_temp2, blocksize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_temp2\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_lambda, blocksize * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_lambda\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramRBR, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramRBR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramPBP, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramPBP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramXAR, blocksize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramXAR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramRAR, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramRAR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_transGramRAR, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_transGramRAR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_identity_PAP, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_identity_PAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_identity_BB, blocksize * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_identity_BB\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_zeros_B_CB, blocksize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_zeros_B_CB\n"); return 0; }

    cuberror = cudaMalloc ((void**)&d_gramXAP, blocksize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramXAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramPAP, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramPAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramRAP, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramRAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramXBP, blocksize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramXBP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramRBP, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramRBP\n"); return 0; }

    cuberror = cudaMalloc ((void**)&d_xrem, nnonzero * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_xrem\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_irem, nnonzero * sizeof(int));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_irem\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_colptrs, (numrows+1) * sizeof(int));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_colptrs\n"); return 0; }

    cuberror = cudaMalloc ((void**)&d_activeMask, blocksize * sizeof(int));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_activeMask\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_residualNorms, blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_residualNorms\n"); return 0; }

    cuberror = cudaMalloc ((void**)&d_gramA, gramASize * gramASize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramA\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_gramB, gramASize * gramASize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_gramB\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_transGramXAR, currentBlockSize * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_transGramXAR\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_transGramXAP, currentBlockSize * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_transGramXAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_transGramRAP, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_transGramRAP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_transGramXBP, currentBlockSize * blocksize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_transGramXBP\n"); return 0; }
    cuberror = cudaMalloc ((void**)&d_transGramRBP, currentBlockSize * currentBlockSize * sizeof(double));
    if( cuberror != 0 ){ printf("cudaMalloc Filed d_transGramRBP\n"); return 0; }
    

    cuberror = cudaMemcpy(d_xrem, acsr, nnonzero * sizeof(double), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed xrem ==> %d\n", cuberror); return 0; }
    cuberror = cudaMemcpy(d_irem, ja, nnonzero * sizeof(int), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed irem ==> %d\n", cuberror); return 0; }
    cuberror = cudaMemcpy(d_colptrs, ia, (numrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed d_colptrs ==> %d\n", cuberror); return 0; }

    /* initialize cusparse library */
    status = cusparseCreate(&cusparseHandle);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("CUSPARSE Library initialization failed");
        return 1;
    }

    /* create and setup matrix descriptor */
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    cubstat = cublasCreate(&handle);
    if( cubstat != CUBLAS_STATUS_SUCCESS ){ printf("HandleCreationFailure\n"); return 0; }
#endif

    //======== creadting device pointers for GPU end ==========//
    #if defined(USE_CUBLAS)
        cuberror = cudaMemcpy(d_identity_PAP, identity_PAP, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        if( cuberror != 0 ){ printf("cudaMemcpy failed identity_PAP ==> %d\n", cuberror); return 0; }
        cuberror = cudaMemcpy(d_identity_BB, identity_BB, blocksize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        if( cuberror != 0 ){ printf("cudaMemcpy failed identity_BB ==> %d\n", cuberror); return 0; }
        cuberror = cudaMemcpy(d_zeros_B_CB, zeros_B_CB, blocksize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
    if( cuberror != 0 ){ printf("cudaMemcpy failed d_zeros_B_CB ==> %d\n", cuberror); return 0; }
        cuberror = cudaMemcpy(d_blockVectorX, blockVectorX, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorX ==> %d\n", cuberror); return 0; }
        cuberror = cudaMemcpy(d_blockVectorAX, blockVectorAX, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorAX ==> %d\n", cuberror); return 0; }
        cuberror = cudaMemcpy(d_blockVectorP, blockVectorP, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorP ==> %d\n", cuberror); return 0; }
        cuberror = cudaMemcpy(d_blockVectorAP, blockVectorAP, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorAP ==> %d\n", cuberror); return 0; }
        cudaDeviceSynchronize();

    #endif
    //loop starts here
    for(iterationNumber = 1 ; iterationNumber <= maxIterations ; iterationNumber++)
    {
        printf("**---------------- iterationNumber : %ld ----------------**\n", iterationNumber);
        // for(i = 0 ; i < numTaks ; i++)
        //     taskTiming[i] = 0;

        //cout << "\niterationNumber: " << iterationNumber << endl;
        loop_start_time = omp_get_wtime();

        //if 12 nested if
        //blockVectorR = blockVectorAX - blockVectorX*spdiags(lambda,0,blockSize,blockSize);
    #if defined(USE_CPU)
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorX, blocksize, lambda,blocksize, 0.0, blockVectorR, blocksize); //XY code 1
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
    #endif 

    #if defined( USE_CUBLAS )
        // cuberror = cudaMemcpy(d_blockVectorX, blockVectorX, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorX ==> %d\n", cuberror); return 0; }
        tstart = omp_get_wtime();
        cubstat = cublasSetMatrix(blocksize, blocksize, sizeof(double), lambda, blocksize, d_lambda, blocksize); //copy_lambda
        if( cubstat != CUBLAS_STATUS_SUCCESS ){ printf("SetMatrixFailure lambda\n"); return 0; }
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, blocksize, 
                  &cudaAlpha, d_lambda, blocksize, d_blockVectorX, blocksize, &cudaBeta, d_blockVectorR, blocksize); //xy_1
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
            printf("cublasDgemm status-1: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
    #endif

        // printf("iterationNumber : %ld blockVectorR ==>\n", iterationNumber);
        // print_mat(blockVectorR, 2, blocksize);
        // printf("\n");
    #if defined(USE_OPENACC)
        // cuberror = cudaMemcpy(d_blockVectorAX, blockVectorAX, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorAX ==> %d\n", cuberror); return 0; }
        tstart = omp_get_wtime();
        mat_sub_OpenACC(d_blockVectorAX, d_blockVectorR, d_blockVectorR , M, blocksize); //sub_1
        taskTiming[iterationNumber - 1][4] += (omp_get_wtime() - tstart); 

        tstart = omp_get_wtime(); 
        #pragma acc parallel loop deviceptr(d_residualNorms)  //setzero(RN)
        for(i = 0 ; i < blocksize ; i++)
            d_residualNorms[i] = 0.0;
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        mat_mult_OpenACC(d_blockVectorR, d_blockVectorR, d_newX, M, blocksize); //mult_1
        taskTiming[iterationNumber - 1][5] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        sum_sqrt_OpenACC(d_newX, d_residualNorms, M, blocksize); //norm
        taskTiming[iterationNumber - 1][16] += (omp_get_wtime() - tstart); 

        // tstart = omp_get_wtime();
        // update_activeMask_OpenACC(d_activeMask, d_residualNorms, residualTolerance, blocksize);
        // taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart); //UPDATE : 8

        // cuberror = cudaMemcpy(blockVectorR, d_blockVectorR, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorR: %d\n", cuberror);}
        tstart = omp_get_wtime();
        cuberror = cudaMemcpy(residualNorms, d_residualNorms, blocksize * sizeof(double), cudaMemcpyDeviceToHost);
        if( cuberror != 0 ){ printf("cudaMemcpy failed activeMask: %d\n", cuberror);}
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);
    #endif

    #if defined(USE_CPU)
        tstart = omp_get_wtime();
        mat_sub(blockVectorAX, d_blockVectorR, d_blockVectorR , M, blocksize); //SUB : 4
        taskTiming[iterationNumber - 1][4] += (omp_get_wtime() - tstart); 
        
        tstart = omp_get_wtime();
        #pragma omp parallel for default(shared)
        for(i = 0 ; i < blocksize ; i++)
            residualNorms[i] = 0.0;
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);
        
        tstart = omp_get_wtime();
        mat_mult(blockVectorR, blockVectorR, newX, M, blocksize); //MULT : 5
        taskTiming[iterationNumber - 1][5] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        sum_sqrt(newX, residualNorms, M, blocksize);
        taskTiming[iterationNumber - 1][16] += (omp_get_wtime() - tstart);

    #endif
        //residualNorms=full(sqrt(sum(conj(blockVectorR).*blockVectorR)')); 
        //residualNormsHistory(1:blockSize,iterationNumber)=residualNorms;
        //activeMask = full(residualNorms > residualTolerance) & activeMask;
        tstart = omp_get_wtime();
        update_activeMask(activeMask, residualNorms, residualTolerance, blocksize);
        taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart); //UPDATE : 8
        //currentBlockSize = sum(activeMask);
        currentBlockSize = 0;
        tstart = omp_get_wtime();
        for(i = 0 ; i < blocksize ; i++)
            currentBlockSize += activeMask[i];
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);

        if(currentBlockSize == 0)
        {
            cout << "converge!!" << endl;
            break;
        }
        //if loop-17
        //blockVectorR(:,activeMask) = blockVectorR(:,activeMask) - ...
        //        blockVectorX*(blockVectorX'*blockVectorR(:,activeMask));
    #if defined(USE_CPU)
        tstart = omp_get_wtime();
        getActiveBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize); //GET: 7
        taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);
    #endif
    #if defined(USE_OPENACC)
        tstart = omp_get_wtime();
        cuberror = cudaMemcpy(d_activeMask, activeMask, blocksize * sizeof(int), cudaMemcpyHostToDevice);
        if( cuberror != 0 ){ printf("cudaMemcpy failed activeMask ==> %d\n", cuberror); return 0; }
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);
        
        tstart = omp_get_wtime();
        getActiveBlockVector_OpenACC(d_activeBlockVectorR, d_activeMask, d_blockVectorR, M, blocksize, currentBlockSize); //GET: 7
        taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);
        
        // cuberror = cudaMemcpy(activeBlockVectorR, d_activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorR: %d\n", cuberror);}
    #endif
        //blockVectorX'*blockVectorR(:,activeMask)  -> temp2 is the result
        // tstart = omp_get_wtime();
        // std::memset(temp2, 0.0, sizeof(temp2));
        // taskTiming[iterationNumber - 1][15] += (omp_get_wtime() - tstart);

    #if defined(USE_BLAS) 
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorX, blocksize, activeBlockVectorR, currentBlockSize,0.0, temp2, currentBlockSize); //XTY : 2
        //_XTY(blockVectorX, activeBlockVectorR, temp2, M, blocksize, currentBlockSize, block_width);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
    #endif

    #if defined( USE_CUBLAS )
        // cuberror = cudaMemcpy(d_blockVectorX, blockVectorX, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorX ==> %d\n", cuberror); return 0; }
        // cuberror = cudaMemcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorX ==> %d\n", cuberror); return 0; }
        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, blocksize, M, 
                  &cudaAlpha, d_activeBlockVectorR, currentBlockSize, d_blockVectorX, blocksize, &cudaBeta, d_temp2, currentBlockSize); //XY code 1
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
            printf("cublasDgemm status-2: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
        //d_temp2 is used in next operation no need to copy id back to host
        // cuberror = cudaMemcpy(temp2, d_temp2, blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed temp2: %d\n", cuberror);}
    #endif
        // printf("iterationNumber : %ld temp2 ==>\n", iterationNumber);
        // print_mat(temp2, 2, currentBlockSize);
        // printf("\n");
        //temp3 = blockVectorX * temp2
        
    #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, blocksize, 1.0, blockVectorX, blocksize, temp2, currentBlockSize, 0.0, temp3, currentBlockSize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
    #endif

    #if defined(USE_CUBLAS)
        //d_blockVectorX and d_temp2 are already in the device mem space, no need to copy them again.
        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, currentBlockSize, M, blocksize, 
                  &cudaAlpha, d_temp2, currentBlockSize, d_blockVectorX, blocksize, &cudaBeta, d_temp3, currentBlockSize); //XY code 1
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
            printf("cublasDgemm status-3: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        //sending d_temp2 and d_temp3 to host mem space.
        // cuberror = cudaMemcpy(temp2, d_temp2, blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed temp2: %d\n", cuberror);}
        
        // cuberror = cudaMemcpy(temp3, d_temp3, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed temp3: %d\n", cuberror);}
    #endif
        // printf("iterationNumber : %ld temp3 ==>\n", iterationNumber);
        // print_mat(temp3, 2, currentBlockSize);
        // printf("\n");
    #if defined(USE_CPU)
        tstart = omp_get_wtime();
        mat_sub(activeBlockVectorR, temp3, activeBlockVectorR, M, currentBlockSize);
        taskTiming[iterationNumber - 1][4] += (omp_get_wtime() - tstart);
    #endif
    #if defined(USE_OPENACC)
        tstart = omp_get_wtime();
        mat_sub_OpenACC(d_activeBlockVectorR, d_temp3, d_activeBlockVectorR, M, currentBlockSize);
        taskTiming[iterationNumber - 1][4] += (omp_get_wtime() - tstart);
        // cuberror = cudaMemcpy(activeBlockVectorR, d_activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorR: %d\n", cuberror);}
        
    #endif
        

        // tstart = omp_get_wtime();
        // updateBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize); //UPDATE: 8
        // taskTiming[8] += (omp_get_wtime() - tstart);
    
        //------- if 18 ------
        //gramRBR=blockVectorR(:,activeMask)'*blockVectorR(:,activeMask);  //blockVectorR(:,activeMask) ->activeBlockVectorR

        
    #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorR, currentBlockSize, activeBlockVectorR, currentBlockSize, 0.0,gramRBR, currentBlockSize);
        //_XTY(activeBlockVectorR, activeBlockVectorR, gramRBR, M, currentBlockSize, currentBlockSize, block_width);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
    #endif
    #if defined( USE_CUBLAS )
        // cuberror = cudaMemcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("==> 634: cudaMemcpy failed activeBlockVectorR ==> %d\n", cuberror); return 0; }
        
        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, currentBlockSize, M, 
                  &cudaAlpha, d_activeBlockVectorR, currentBlockSize, d_activeBlockVectorR, currentBlockSize, &cudaBeta, d_gramRBR, currentBlockSize); //XY code 1
        if(cubstat != CUBLAS_STATUS_SUCCESS)
            printf("cublasDgemm status-4: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        cuberror = cudaMemcpy(gramRBR, d_gramRBR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        if( cuberror != 0 ){ printf("cudaMemcpy failed d_gramRBR: %d\n", cuberror);}
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);
    #endif

        //[gramRBR,cholFlag]=chol(gramRBR);
        // tstart = omp_get_wtime();
        // transpose(gramRBR, trans_gramRBR, currentBlockSize, currentBlockSize);
        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
        
        // tstart = omp_get_wtime();
        // dpotrf_( &uplo, &currentBlockSize, trans_gramRBR, &currentBlockSize, &info );
        // taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);
        
        // if(info != 0)
        // {
        //     cout<<"dportf_ error 2!!"<<endl;
        //     break;
        // }
        // tstart = omp_get_wtime();
        // transpose(trans_gramRBR, gramRBR, currentBlockSize, currentBlockSize);
        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR , 'U' , currentBlockSize , gramRBR , currentBlockSize);
        taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);

        if(info != 0)
            cout << "dportf_ error 2!!" << endl;

        tstart = omp_get_wtime();
        #pragma omp parallel for private(j) default(shared)
        for(i = 0 ; i < currentBlockSize ; i++)
        {
            for(j = 0 ; j < i ; j++)
            {
                gramRBR[i * currentBlockSize + j] = 0.0;
            }
        }
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);
        //------- if 18 nested if -----
        if(info == 0)
        {
            tstart = omp_get_wtime();
            inverse(gramRBR, currentBlockSize, currentBlockSize);
            taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
        
        #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorR, currentBlockSize, gramRBR, currentBlockSize, 0.0, temp3, currentBlockSize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif

        #if defined(USE_CUBLAS)
            //d_activeBlockVectorR is already in device memory and activeBlockVectorR has not been changed since then.
            tstart = omp_get_wtime();
            cuberror = cudaMemcpy(d_gramRBR, gramRBR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
            if( cuberror != 0 ){ printf("cudaMemcpy failed d_gramRBR ==> %d\n", cuberror); return 0; }
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);

            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, currentBlockSize, M, currentBlockSize, 
                    &cudaAlpha, d_gramRBR, currentBlockSize, d_activeBlockVectorR, currentBlockSize, &cudaBeta, d_temp3, currentBlockSize); //XY code 1
            
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                printf("cublasDgemm status-5: %d\n",cubstat);
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            //cuberror = cudaMemcpy(temp3, d_temp3, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
            //if( cuberror != 0 ){ printf("cudaMemcpy failed temp3: %d\n", cuberror);}
        #endif
        #if defined(USE_CPU)
            tstart = omp_get_wtime();
            custom_dlacpy(temp3, activeBlockVectorR, M, currentBlockSize); //DLACPY: 11
            taskTiming[iterationNumber - 1][10] += (omp_get_wtime() - tstart);

            tstart = omp_get_wtime();
            updateBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize);
            taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);
        #endif
        #if defined(USE_OPENACC)
            tstart = omp_get_wtime();
            custom_dlacpy_OpenACC(d_temp3, d_activeBlockVectorR, M, currentBlockSize); //DLACPY: 11
            taskTiming[iterationNumber - 1][10] += (omp_get_wtime() - tstart);
        
            tstart = omp_get_wtime();
            updateBlockVector_OpenACC(d_activeBlockVectorR, d_activeMask, d_blockVectorR, M, blocksize, currentBlockSize);
            taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);

            // cuberror = cudaMemcpy(activeBlockVectorR, d_activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed d_activeBlockVectorR: %d\n", cuberror);}
            // cuberror = cudaMemcpy(blockVectorR, d_blockVectorR, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorR: %d\n", cuberror);}
        #endif
            // printf("iterationNumber : %ld activeBlockVectorR ==>\n", iterationNumber);
            // print_mat(activeBlockVectorR, 2, currentBlockSize);
            // printf("\n");
            
            // printf("iterationNumber : %ld blockVectorR ==>\n", iterationNumber);
            // print_mat(blockVectorR, 2, blocksize);
            // printf("\n");
        } //end if

    #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        #pragma omp parallel for private(j) default(shared)
        for(i = 0; i < M ; i++)
        {
            for(j = 0 ; j < currentBlockSize ; j++)
            {
                activeBlockVectorAR[i * currentBlockSize + j] = 0.0;
            }
        }
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart); //SETZERO : 0
        tstart = omp_get_wtime();
        //mkl_dcscmm(&transA, &M, &currentBlockSize, &M, &alpha, matdescra, xrem, irem, colptrs, colptrs+1, activeBlockVectorR, &currentBlockSize,  &beta, activeBlockVectorAR, &currentBlockSize);
        //mkl_dcsrmm(&transA, &M, &currentBlockSize, &M, &alpha, matdescra, acsr, ja, ia, ia+1, activeBlockVectorR, &currentBlockSize, &beta, activeBlockVectorAR, &currentBlockSize);
        spmm_csr(numrows, numcols, currentBlockSize, ia, ja, acsr, activeBlockVectorR, activeBlockVectorAR);
        taskTiming[iterationNumber - 1][6] += (omp_get_wtime() - tstart); //SPMM 6
    #endif
        
    #if defined(USE_CUBLAS)
        tstart = omp_get_wtime();
        //transpose(activeBlockVectorR, newX, currentBlockSize, M);
        transpose_OpenACC(d_activeBlockVectorR, d_newX, currentBlockSize, M);
        taskTiming[iterationNumber - 1][6] += (omp_get_wtime() - tstart);

        // cuberror = cudaMemcpy(d_activeBlockVectorR, newX, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorR ==> %d\n", cuberror); return 0; }
        
        // tstart = omp_get_wtime();
        // custom_dlacpy_OpenACC(d_newX, d_activeBlockVectorR, M, currentBlockSize); //DLACPY: 11
        // taskTiming[iterationNumber - 1][10] += (omp_get_wtime() - tstart);

        // cudaError_t cudaStat1 = cudaMemset((void *)d_activeBlockVectorAR, 0, M * currentBlockSize * sizeof(double));
        // if (cudaStat1 != cudaSuccess) { printf("Memset on Device failed"); return 1; }
        
        tstart = omp_get_wtime();
        status = cusparseDcsrmm(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, currentBlockSize, M,
                            nnonzero, &cudaAlpha, descr, d_xrem, d_colptrs, d_irem,
                            d_newX, M, &cudaBeta, d_temp3, M);
        
        if (status != CUSPARSE_STATUS_SUCCESS) 
            printf("cusparseDcsrmm status: %d\n", status);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][6] += (omp_get_wtime() - tstart);
        //printf("SpMM status: %d\n", status);

        // cuberror = cudaMemcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorR ==> %d\n", cuberror); return 0; }

        tstart = omp_get_wtime();
        transpose_OpenACC(d_temp3, d_activeBlockVectorAR, M, currentBlockSize);
        taskTiming[iterationNumber - 1][6] += (omp_get_wtime() - tstart);

        // cuberror = cudaMemcpy(activeBlockVectorAR, d_activeBlockVectorAR, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorAR: %d\n", cuberror);}
    #endif
        

        if(iterationNumber > 1)
        {
            //if 20 first nested if
            // gramPBP=blockVectorP(:,activeMask)'*blockVectorP(:,activeMask);
        #if defined(USE_CPU)
            tstart = omp_get_wtime();
            getActiveBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);
            taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);
        #endif
        
        #if defined(USE_OPENACC)
            // cuberror = cudaMemcpy(d_blockVectorP, blockVectorP, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorP ==> %d\n", cuberror); return 0; }
            
            tstart = omp_get_wtime();
            getActiveBlockVector_OpenACC(d_activeBlockVectorP, d_activeMask, d_blockVectorP, M, blocksize, currentBlockSize);
            taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);
            
            // cuberror = cudaMemcpy(activeBlockVectorP, d_activeBlockVectorP, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorP: %d\n", cuberror);}
        #endif

        #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorP, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramPBP, currentBlockSize);
            //_XTY(activeBlockVectorP, activeBlockVectorP, gramPBP, M, currentBlockSize, currentBlockSize, block_width);
            taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
        #endif

        #if defined( USE_CUBLAS )
            //cuberror = cudaMemcpy(d_activeBlockVectorP, activeBlockVectorP, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
            //if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorP ==> %d\n", cuberror); return 0; }
            
            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, currentBlockSize, M, 
                    &cudaAlpha, d_activeBlockVectorP, currentBlockSize, d_activeBlockVectorP, currentBlockSize, &cudaBeta, d_gramPBP, currentBlockSize); //XY 
            
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                printf("cublasDgemm status-6: %d\n",cubstat);
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

            tstart = omp_get_wtime();
            cuberror = cudaMemcpy(gramPBP, d_gramPBP, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
            if( cuberror != 0 ){ printf("cudaMemcpy failed d_gramRBR: %d\n", cuberror);}
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);
        #endif

            // tstart = omp_get_wtime();
            // transpose(gramPBP, trans_gramPBP, currentBlockSize, currentBlockSize);
            // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
            
            // tstart = omp_get_wtime();
            // dpotrf_( &uplo, &currentBlockSize, trans_gramPBP, &currentBlockSize, &info );
            // taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);

            // if(info != 0)
            // {
            //     cout<<"dportf_ error 3"<<endl;
            //     break;
            // }
            // tstart = omp_get_wtime();
            // transpose(trans_gramPBP, gramPBP, currentBlockSize, currentBlockSize);
            // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

            tstart = omp_get_wtime();
            info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR , 'U' , currentBlockSize , gramPBP , currentBlockSize);
            taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);
            
            if(info != 0)
                cout<<"dportf_ error 3"<<endl;
             
            //making the lower part of gramPBP zero
            tstart = omp_get_wtime();
            #pragma omp parallel for private(j) default(shared)
            for(i = 0 ; i < currentBlockSize ; i++)
            {
                for(j = 0 ; j < i ; j++)
                {
                    gramPBP[i * currentBlockSize + j] = 0.0;
                }
            }
            taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);

            if(info == 0)
            {
                //if 20 first nested if 2
                // blockVectorP(:,activeMask) = blockVectorP(:,activeMask)/gramPBP;
                tstart = omp_get_wtime();
                inverse(gramPBP, currentBlockSize, currentBlockSize);
                taskTiming[iterationNumber - 1][11] += (omp_get_wtime() - tstart);
            
            #if defined(USE_BLAS)
                tstart = omp_get_wtime();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorP, currentBlockSize, gramPBP, currentBlockSize, 0.0, temp3, currentBlockSize);
                taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            #endif

            #if defined(USE_CUBLAS)
                //d_activeBlockVectorP is already in the device mem space
                tstart = omp_get_wtime();
                cuberror = cudaMemcpy(d_gramPBP, gramPBP, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                if( cuberror != 0 ){ printf("cudaMemcpy failed gramPBP ==> %d\n", cuberror); return 0; }
                cudaDeviceSynchronize();
                taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);

                tstart = omp_get_wtime();
                cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, currentBlockSize, M, currentBlockSize, 
                        &cudaAlpha, d_gramPBP, currentBlockSize, d_activeBlockVectorP, currentBlockSize, &cudaBeta, d_temp3, currentBlockSize);  
                
                if(cubstat != CUBLAS_STATUS_SUCCESS)
                    printf("cublasDgemm status-7: %d\n",cubstat);
                cudaDeviceSynchronize();
                taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);

                //cuberror = cudaMemcpy(temp3, d_temp3, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                //if( cuberror != 0 ){ printf("cudaMemcpy failed d_temp3: %d\n", cuberror);}
            #endif
            #if defined(USE_CPU)
                tstart = omp_get_wtime();
                custom_dlacpy(temp3, activeBlockVectorP, M, currentBlockSize);
                taskTiming[iterationNumber -1][10] += (omp_get_wtime() - tstart);

                tstart = omp_get_wtime();
                updateBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);

            #endif
            #if defined(USE_OPENACC)
                tstart = omp_get_wtime();
                custom_dlacpy_OpenACC(d_temp3, d_activeBlockVectorP, M, currentBlockSize);
                taskTiming[iterationNumber -1][10] += (omp_get_wtime() - tstart);

                tstart = omp_get_wtime();
                updateBlockVector_OpenACC(d_activeBlockVectorP, d_activeMask, d_blockVectorP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);

                // cuberror = cudaMemcpy(activeBlockVectorP, d_activeBlockVectorP, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorP: %d\n", cuberror);}

                // cuberror = cudaMemcpy(blockVectorP, d_blockVectorP, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
                // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorP: %d\n", cuberror);}
            #endif
            
            #if defined(USE_CPU)
                //blockVectorAP(:,activeMask) = blockVectorAP(:,activeMask)/gramPBP;
                tstart = omp_get_wtime();
                getActiveBlockVector(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);
            #endif
            
            #if defined(USE_OPENACC)
                // cuberror = cudaMemcpy(d_blockVectorAP, blockVectorAP, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorAP ==> %d\n", cuberror); return 0; }
                
                tstart = omp_get_wtime();
                getActiveBlockVector_OpenACC(d_activeBlockVectorAP, d_activeMask, d_blockVectorAP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);

                // cuberror = cudaMemcpy(activeBlockVectorAP, d_activeBlockVectorAP, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorAP: %d\n", cuberror);}

            #endif

            #if defined(USE_BLAS)
                tstart = omp_get_wtime();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorAP, currentBlockSize, gramPBP, currentBlockSize, 0.0, temp3, currentBlockSize);
                taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            #endif 

            #if defined( USE_CUBLAS )
                //d_gramPBP is already in the device mem space
                //cuberror = cudaMemcpy(d_activeBlockVectorAP, activeBlockVectorAP, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                //if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorAP ==> %d\n", cuberror); return 0; }
                
                tstart = omp_get_wtime();
                cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, currentBlockSize, M, currentBlockSize, 
                        &cudaAlpha, d_gramPBP, currentBlockSize, d_activeBlockVectorAP, currentBlockSize, &cudaBeta, d_temp3, currentBlockSize);  
                
                if(cubstat != CUBLAS_STATUS_SUCCESS)
                    printf("cublasDgemm status-8: %d\n",cubstat);
                cudaDeviceSynchronize();
                taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
                //cuberror = cudaMemcpy(temp3, d_temp3, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                //if( cuberror != 0 ){ printf("cudaMemcpy failed d_temp3: %d\n", cuberror);}
            #endif
            #if defined(USE_CPU)
                tstart = omp_get_wtime();
                custom_dlacpy(temp3, activeBlockVectorAP, M, currentBlockSize);
                taskTiming[iterationNumber - 1][10] += (omp_get_wtime() - tstart);

                tstart = omp_get_wtime();
                updateBlockVector(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);
            #endif
            #if defined(USE_OPENACC)
                tstart = omp_get_wtime();
                custom_dlacpy_OpenACC(d_temp3, d_activeBlockVectorAP, M, currentBlockSize);
                taskTiming[iterationNumber - 1][10] += (omp_get_wtime() - tstart);
                
                tstart = omp_get_wtime();
                updateBlockVector_OpenACC(d_activeBlockVectorAP, d_activeMask, d_blockVectorAP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);

                // cuberror = cudaMemcpy(activeBlockVectorAP, d_activeBlockVectorAP, M * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorAP: %d\n", cuberror);}

                // cuberror = cudaMemcpy(blockVectorAP, d_blockVectorAP, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
                // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorAP: %d\n", cuberror);}

            #endif
                
            } //end if info
            else
            {
                cout << "BLOPEX:lobpcg:DirectionNotFullRank...The direction matrix is not full rank." << endl;
            }
        } //end outer if

        //restart=1;
        //The Raileight-Ritz method for [blockVectorX blockVectorR blockVectorP]

        //------ if 21
        int flag = 1;
        tstart = omp_get_wtime();
        for(i = 0 ; i < blocksize ; i++)
        {
            //cout<<"residualNorms[i] :"<<residualNorms[i]<<endl;
            if(residualNorms[i] < 4.0538e-10)
            {
                flag = 0;
                break;
            }
        }
        if(flag == 0)
            explicitGramFlag = 1;
        else
            explicitGramFlag = 0;

        activeRSize = currentBlockSize;

        //---- if 22 -----
        //cout<<"if 22"<<endl;
        if(iterationNumber == 1)
        {
            activePSize = 0;
            restart = 1;
            //cout<<"restart: "<<restart<<endl;
        }
        else
        {
            activePSize = currentBlockSize;
            restart = 0;
        }
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);
        //gramXAR=full(blockVectorAX'*blockVectorR(:,activeMask));
        //gramRAR=full(blockVectorAR(:,activeMask)'*blockVectorR(:,activeMask));
        //gramRAR=(gramRAR'+gramRAR)*0.5;
    #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorAX, blocksize, activeBlockVectorR, currentBlockSize, 0.0, gramXAR, currentBlockSize);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
    #endif

    #if defined( USE_CUBLAS )
        // cuberror = cudaMemcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("==> 920: cudaMemcpy failed activeBlockVectorR ==> %d\n", cuberror); return 0; }
        // cuberror = cudaMemcpy(d_blockVectorAX, blockVectorAX, M * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed blockVectorAX ==> %d\n", cuberror); return 0; }
        
        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, blocksize, M, 
                &cudaAlpha, d_activeBlockVectorR, currentBlockSize, d_blockVectorAX, blocksize, &cudaBeta, d_gramXAR, currentBlockSize); //XY 
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
                    printf("cublasDgemm status-9: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
        // cuberror = cudaMemcpy(gramXAR, d_gramXAR, blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed d_gramXAR: %d\n", cuberror);}
    #endif
    
    #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAR, currentBlockSize, activeBlockVectorR, currentBlockSize, 0.0, gramRAR, currentBlockSize);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
    #endif

    #if defined( USE_CUBLAS )
        //d_activeBlockVectorR := already in the device mem. space 
        // cuberror = cudaMemcpy(d_activeBlockVectorAR, activeBlockVectorAR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorAR ==> %d\n", cuberror); return 0; }
        
        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, currentBlockSize, M, 
                &cudaAlpha, d_activeBlockVectorR, currentBlockSize, d_activeBlockVectorAR, currentBlockSize, &cudaBeta, d_gramRAR, currentBlockSize); //XY 
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
                    printf("cublasDgemm status-10: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
        // cuberror = cudaMemcpy(gramRAR, d_gramRAR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed d_gramRAR: %d\n", cuberror);}
    #endif

    #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize);
        taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, transGramRAR, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramRAR, currentBlockSize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
    #endif

    #if defined(USE_CUBLAS)
        //d_gramRAR is already in the device mem space
        double cudaAlphaHalf = 0.5, cudaBetaHalf = 0.5;

        tstart = omp_get_wtime();
        transpose_OpenACC(d_gramRAR, d_transGramRAR, currentBlockSize, currentBlockSize);
        taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

        // cuberror = cudaMemcpy(d_transGramRAR, transGramRAR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed transGramRAR ==> %d\n", cuberror); return 0; }
        // cuberror = cudaMemcpy(d_identity_PAP, identity_PAP, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed identity_PAP ==> %d\n", cuberror); return 0; }
        
        tstart = omp_get_wtime();     
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, currentBlockSize, currentBlockSize, currentBlockSize, 
                &cudaAlphaHalf, d_identity_PAP, currentBlockSize, d_transGramRAR, currentBlockSize, &cudaBetaHalf, d_gramRAR, currentBlockSize);  
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
                    printf("cublasDgemm status-11: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        // cuberror = cudaMemcpy(gramRAR, d_gramRAR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed d_gramRAR: %d\n", cuberror);}

    #endif
     
        //--- cond_try for loop -----
        for(int cond_try = 1 ; cond_try <=2 ; cond_try++)
        {
            if(restart == 0) //---- if 24 ----
            {
                if(restart == 0)
                {
                    //gramXAP=full(blockVectorAX'*blockVectorP(:,activeMask));
                #if defined(USE_BLAS)
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorAX, blocksize, activeBlockVectorP, currentBlockSize, 0.0, gramXAP, currentBlockSize);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                #endif

                #if defined( USE_CUBLAS )
                    //d_blockVectorAX is already in the device mem space
                    // cuberror = cudaMemcpy(d_activeBlockVectorP, activeBlockVectorP, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                    // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorP ==> %d\n", cuberror); return 0; }

                    tstart = omp_get_wtime();       
                    cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, blocksize, M, 
                            &cudaAlpha, d_activeBlockVectorP, currentBlockSize, d_blockVectorAX, blocksize, &cudaBeta, d_gramXAP, currentBlockSize);  
                    
                    if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-12: %d\n",cubstat);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

                    // cuberror = cudaMemcpy(gramXAP, d_gramXAP, blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                    // if( cuberror != 0 ){ printf("cudaMemcpy failed gramXAP: %d\n", cuberror);}
                #endif
                
                #if defined(USE_BLAS)
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAR, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramRAP, currentBlockSize);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                #endif

                #if defined( USE_CUBLAS )
                    //d_activeBlockVectorP & d_activeBlockVectorAR are already in the device mem space
                    tstart = omp_get_wtime();
                    cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, currentBlockSize, M, 
                            &cudaAlpha, d_activeBlockVectorP, currentBlockSize, d_activeBlockVectorAR, currentBlockSize, &cudaBeta, d_gramRAP, currentBlockSize);  
                    
                    if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-13: %d\n",cubstat);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

                    // cuberror = cudaMemcpy(gramRAP, d_gramRAP, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                    // if( cuberror != 0 ){ printf("cudaMemcpy failed gramRAP: %d\n", cuberror);}
                #endif
                
                #if defined(USE_BLAS)
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAP, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramPAP, currentBlockSize);
                    //_XTY(activeBlockVectorAP, activeBlockVectorP, gramPAP, M, currentBlockSize, currentBlockSize, block_width);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                #endif

                #if defined( USE_CUBLAS )
                    //d_activeBlockVectorP is already in the device mem space
                    // cuberror = cudaMemcpy(d_activeBlockVectorAP, activeBlockVectorAP, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                    // if( cuberror != 0 ){ printf("cudaMemcpy failed activeBlockVectorAP ==> %d\n", cuberror); return 0; }
                    
                    tstart = omp_get_wtime();    
                    cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, currentBlockSize, M, 
                            &cudaAlpha, d_activeBlockVectorP, currentBlockSize, d_activeBlockVectorAP, currentBlockSize, &cudaBeta, d_gramPAP, currentBlockSize);  
                    
                    if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-14: %d\n",cubstat);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                    //d_gramPAP is used in the next operation
                #endif

                    //gramPAP=(gramPAP'+gramPAP)*0.5;
                #if defined(USE_BLAS)
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, gramPAP, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramPAP, currentBlockSize);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                #endif

                #if defined( USE_CUBLAS )
                    //d_gramPAP is in the device memory 
                    tstart = omp_get_wtime();
                    cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, currentBlockSize, currentBlockSize, 
                            &cudaAlphaHalf, d_identity_PAP, currentBlockSize, d_gramPAP, currentBlockSize, &cudaBetaHalf, d_gramPAP, currentBlockSize); 
                    
                    if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-15: %d\n",cubstat);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart); 
                    // cuberror = cudaMemcpy(gramPAP, d_gramPAP, blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                    // if( cuberror != 0 ){ printf("cudaMemcpy failed gramPAP: %d\n", cuberror);}
                #endif

                    if(explicitGramFlag == 1)
                    {
                        cout<<"nested if 24"<<endl;
                    }
                    else
                    {
                        //cout<<"if 24 nested if 1 else"<<endl;
                        //gramA = [ diag(lambda)  gramXAR  gramXAP
                        //           gramXAR'      gramRAR  gramRAP
                        //           gramXAP'      gramRAP'  gramPAP ];
                        cudaDeviceSynchronize();
                        gramASize = blocksize + currentBlockSize + currentBlockSize;

                        //gramA = new double[gramASize * gramASize]();

                        #if defined(USE_CPU)
                            tstart = omp_get_wtime();
                            mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                            mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                            mat_copy(gramXAP, blocksize, currentBlockSize, gramA, 0, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                            taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                            mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                            mat_copy(gramRAP, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            transpose(gramXAP, transGramXAP, currentBlockSize, blocksize); 
                            transpose(gramRAP, transGramRAP, currentBlockSize, currentBlockSize);
                            taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            mat_copy(transGramXAP, currentBlockSize, blocksize, gramA, blocksize+currentBlockSize, 0, gramASize);
                            mat_copy(transGramRAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize, gramASize);
                            mat_copy(gramPAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                        #endif

                        #if defined(USE_OPENACC)
                            tstart = omp_get_wtime();
                            mat_copy_OpenACC(d_lambda, blocksize, blocksize, d_gramA, 0, 0, gramASize);
                            mat_copy_OpenACC(d_gramXAR, blocksize, currentBlockSize, d_gramA, 0, blocksize, gramASize);
                            mat_copy_OpenACC(d_gramXAP, blocksize, currentBlockSize, d_gramA, 0, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            transpose_OpenACC(d_gramXAR, d_transGramXAR, currentBlockSize, blocksize);
                            taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
                            mat_copy_OpenACC(d_transGramXAR, currentBlockSize, blocksize, d_gramA, blocksize, 0, gramASize);
                            
                            tstart = omp_get_wtime();
                            mat_copy_OpenACC(d_gramRAR, currentBlockSize, currentBlockSize, d_gramA, blocksize, blocksize, gramASize);
                            mat_copy_OpenACC(d_gramRAP, currentBlockSize, currentBlockSize, d_gramA, blocksize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            transpose_OpenACC(d_gramXAP, d_transGramXAP, currentBlockSize, blocksize); 
                            transpose_OpenACC(d_gramRAP, d_transGramRAP, currentBlockSize, currentBlockSize);
                            taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            mat_copy_OpenACC(d_transGramXAP, currentBlockSize, blocksize, d_gramA, blocksize+currentBlockSize, 0, gramASize);
                            mat_copy_OpenACC(d_transGramRAP, currentBlockSize, currentBlockSize, d_gramA, blocksize+currentBlockSize, blocksize, gramASize);
                            mat_copy_OpenACC(d_gramPAP, currentBlockSize, currentBlockSize, d_gramA, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                            
                            tstart = omp_get_wtime();
                            cuberror = cudaMemcpy(gramA, d_gramA, gramASize * gramASize * sizeof(double), cudaMemcpyDeviceToHost);
                            if( cuberror != 0 ){ printf("cudaMemcpy failed gramA: %d\n", cuberror);}
                            taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);
                        #endif
                        
                        // tstart = omp_get_wtime();
                        // mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                        // mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                        // mat_copy(gramXAP, blocksize, currentBlockSize, gramA, 0, blocksize+currentBlockSize, gramASize);
                        // taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                        
                        // tstart = omp_get_wtime();
                        // transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
                        
                        // tstart = omp_get_wtime();
                        // mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                        // mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                        // mat_copy(gramRAP, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize+currentBlockSize, gramASize);
                        // taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                        // tstart = omp_get_wtime();
                        // transpose(gramXAP, transGramXAP, currentBlockSize, blocksize); 
                        // transpose(gramRAP, transGramRAP, currentBlockSize, currentBlockSize);
                        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

                        // tstart = omp_get_wtime();
                        // mat_copy(transGramXAP, currentBlockSize, blocksize, gramA, blocksize+currentBlockSize, 0, gramASize);
                        // mat_copy(transGramRAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize, gramASize);
                        // mat_copy(gramPAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);
                        // taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                    } //end else
                    
                #if defined(USE_BLAS)
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorX, blocksize, activeBlockVectorP, currentBlockSize, 0.0, gramXBP, currentBlockSize);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                #endif

                #if defined( USE_CUBLAS )
                    //d_gramPAP is in the device memory 
                    tstart = omp_get_wtime();
                    cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, blocksize, M, 
                            &cudaAlpha, d_activeBlockVectorP, currentBlockSize, d_blockVectorX, blocksize, &cudaBeta, d_gramXBP, currentBlockSize);  
                    
                    if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-16: %d\n",cubstat);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                    // cuberror = cudaMemcpy(gramXBP, d_gramXBP, blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                    // if( cuberror != 0 ){ printf("cudaMemcpy failed gramXBP: %d\n", cuberror);}
                    // cudaDeviceSynchronize();
                #endif

                #if defined(USE_BLAS)
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorR, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramRBP, currentBlockSize);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                #endif

                #if defined( USE_CUBLAS )
                    //d_gramPAP is in the device memory 
                    tstart = omp_get_wtime();
                    cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, currentBlockSize, M, 
                            &cudaAlpha, d_activeBlockVectorP, currentBlockSize, d_activeBlockVectorR, currentBlockSize, &cudaBeta, d_gramRBP, currentBlockSize);  
                     
                    if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-17: %d\n",cubstat);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                    // cuberror = cudaMemcpy(gramRBP, d_gramRBP, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                    // if( cuberror != 0 ){ printf("cudaMemcpy failed gramXBP: %d\n", cuberror);}
                #endif

                    if(explicitGramFlag==1)
                    {
                        cout << "if 24 nested if 3" << endl;
                    }
                    else
                    {
                        //cout<<"if 24 nested if 3 else"<<endl;
                        //gramB=[eye(blockSize) zeros(blockSize,activeRSize) gramXBP
                        //       zeros(blockSize,activeRSize)' eye(activeRSize) gramRBP
                        //       gramXBP' gramRBP' eye(activePSize) ];

                        //gramB = new double[gramASize * gramASize];
                        cudaDeviceSynchronize();
                        #if defined(USE_CPU)
                            tstart = omp_get_wtime();
                            mat_copy(identity_BB, blocksize, blocksize, gramB, 0, 0, gramASize);
                            mat_copy(zeros_B_CB, blocksize, currentBlockSize, gramB, 0, blocksize, gramASize);
                            mat_copy(gramXBP, blocksize, currentBlockSize, gramB, 0, blocksize+currentBlockSize, gramASize);
                            mat_copy(zeros_CB_B, activeRSize, blocksize, gramB, blocksize, 0, gramASize);
                            mat_copy(identity_PAP, activeRSize, activeRSize, gramB, blocksize, blocksize, gramASize);
                            mat_copy(gramRBP, currentBlockSize, currentBlockSize, gramB, blocksize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                            tstart = omp_get_wtime();
                            transpose(gramXBP, transGramXBP, currentBlockSize, blocksize);
                            transpose(gramRBP, transGramRBP, currentBlockSize, currentBlockSize);
                            taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            mat_copy(transGramXBP, currentBlockSize, blocksize, gramB, blocksize+currentBlockSize, 0, gramASize);
                            mat_copy(transGramRBP, currentBlockSize, currentBlockSize, gramB, blocksize+currentBlockSize, blocksize, gramASize);
                            mat_copy(identity_PAP, activePSize, activePSize, gramB, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                        #endif
                        #if defined(USE_OPENACC)
                            tstart = omp_get_wtime();
                            mat_copy_OpenACC(d_identity_BB, blocksize, blocksize, d_gramB, 0, 0, gramASize);
                            mat_copy_OpenACC(d_zeros_B_CB, blocksize, currentBlockSize, d_gramB, 0, blocksize, gramASize);
                            mat_copy_OpenACC(d_gramXBP, blocksize, currentBlockSize, d_gramB, 0, blocksize+currentBlockSize, gramASize);
                            mat_copy_OpenACC(d_zeros_B_CB, activeRSize, blocksize, d_gramB, blocksize, 0, gramASize);
                            mat_copy_OpenACC(d_identity_PAP, activeRSize, activeRSize, d_gramB, blocksize, blocksize, gramASize);
                            mat_copy_OpenACC(d_gramRBP, currentBlockSize, currentBlockSize, d_gramB, blocksize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            transpose_OpenACC(d_gramXBP, d_transGramXBP, currentBlockSize, blocksize);
                            transpose_OpenACC(d_gramRBP, d_transGramRBP, currentBlockSize, currentBlockSize);
                            taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            mat_copy_OpenACC(d_transGramXBP, currentBlockSize, blocksize, d_gramB, blocksize+currentBlockSize, 0, gramASize);
                            mat_copy_OpenACC(d_transGramRBP, currentBlockSize, currentBlockSize, d_gramB, blocksize+currentBlockSize, blocksize, gramASize);
                            mat_copy_OpenACC(d_identity_PAP, activePSize, activePSize, d_gramB, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);
                            taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                            tstart = omp_get_wtime();
                            cuberror = cudaMemcpy(gramB, d_gramB, gramASize * gramASize * sizeof(double), cudaMemcpyDeviceToHost);
                            if( cuberror != 0 ){ printf("cudaMemcpy failed gramB: %d\n", cuberror);}
                            taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);
                        #endif


                       
                    }
                } //inner if end
            } //outer if end
            else //---- if 24 else ----
            {
                if(explicitGramFlag == 1 ) //--- if 24 else nested if --
                {
                    //cout<<"if 24 else nested if"<<endl;
                    //gramA = [ gramXAX   gramXAR
                    //          gramXAR'    gramRAR  ];
                    //gramB = [ gramXBX  gramXBR
                    //            gramXBR' eye(activeRSize)  ];
                }
                else //--- if 24 else nested else;
                {
                    //cout<<"if 24 else nested else"<<endl;
                    //gramA = [ diag(lambda)  gramXAR
                    //          gramXAR'        gramRAR  ];
                    //gramB = eye(blockSize+activeRSize);
                    cudaDeviceSynchronize();
                    gramASize = blocksize + activeRSize;
                    //gramA = new double[gramASize * gramASize]();
                    #if defined(USE_CPU)
                        tstart = omp_get_wtime();
                        mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                        mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                        taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                        tstart = omp_get_wtime();
                        transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                        taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
                        
                        tstart = omp_get_wtime();
                        mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                        mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                    #endif

                    #if defined(USE_OPENACC)
                        tstart = omp_get_wtime();
                        mat_copy_OpenACC(d_lambda, blocksize, blocksize, d_gramA, 0, 0, gramASize);
                        mat_copy_OpenACC(d_gramXAR, blocksize, currentBlockSize, d_gramA, 0, blocksize, gramASize);
                        taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);

                        tstart = omp_get_wtime();
                        transpose_OpenACC(d_gramXAR, d_transGramXAR, currentBlockSize, blocksize);
                        taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
                        
                        tstart = omp_get_wtime();
                        mat_copy_OpenACC(d_transGramXAR, currentBlockSize, blocksize, d_gramA, blocksize, 0, gramASize);
                        mat_copy_OpenACC(d_gramRAR, currentBlockSize, currentBlockSize, d_gramA, blocksize, blocksize, gramASize);
                        taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                            
                        tstart = omp_get_wtime();
                        cuberror = cudaMemcpy(gramA, d_gramA, gramASize * gramASize * sizeof(double), cudaMemcpyDeviceToHost);
                        if( cuberror != 0 ){ printf("cudaMemcpy failed gramA: %d\n", cuberror);}
                        taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);
                    #endif
                    //gramB = new double[gramASize * gramASize]();
                    tstart = omp_get_wtime();
                    make_identity_mat(gramB, gramASize, gramASize);
                    taskTiming[iterationNumber - 1][13] += (omp_get_wtime() - tstart);
                }
            } //end else

            //--------- if 25 part of it ------
            //cout<<"if 25 part of it"<<endl;
            if(cond_try == 1 && ~restart)
            {
                //cout<<"if 25 else-> break from here"<<endl;
                //restart=1;
                break;
            }
        }//inner loop finish here


        //tstart = omp_get_wtime();
        cudaDeviceSynchronize();
        eigen_value = new double[gramASize]();
        tstart = omp_get_wtime();
        info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, gramASize, gramA, gramASize, gramB, gramASize, eigen_value);
        taskTiming[iterationNumber - 1][9] += (omp_get_wtime() - tstart);

        // double *trans_gramA = new double[gramASize * gramASize]();
        // double *trans_gramB = new double[gramASize * gramASize]();
        
        // tstart = omp_get_wtime();
        // transpose(gramA, trans_gramA, gramASize, gramASize);
        // transpose(gramB, trans_gramB, gramASize, gramASize);
        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

        // tstart = omp_get_wtime();
        // lwork = -1;
        // dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, &work_query, &lwork, &info);
      
        // if(info != 0)
        //     cout<<"Error in dummy call"<<endl;

        // lwork = (int) work_query;

        // work = new double[lwork]();
        // dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, work, &lwork, &info);
        // taskTiming[iterationNumber - 1][9] += (omp_get_wtime() - tstart);

        // if(info != 0)
        //     cout<<"Error in eigen value calculation"<<endl;

        // tstart = omp_get_wtime();
        // transpose(trans_gramA, gramA, gramASize, gramASize);
        // transpose(trans_gramB, gramB, gramASize, gramASize);
        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

        // delete []trans_gramA;
        // delete []trans_gramB;
      
        if( info != 0 )
        { 
            printf("LAPACKE_dsygv error: The algorithm failed to compute eigenvalues.\n" );
            break;
        }

        tstart = omp_get_wtime();
        diag(eigen_value, lambda, blocksize);
        taskTiming[iterationNumber - 1][17] += (omp_get_wtime() - tstart);

        int column = 0;
        coordX = new double[gramASize * blocksize]();
        tstart = omp_get_wtime();
        for(j = 0 ; column < blocksize && j < gramASize ; j++)
        {   
            // #pragma omp parallel for default(shared)
            for(i = 0 ; i < gramASize ; i++)
            {
                coordX[i * blocksize + column] = gramA[i * gramASize + j];
            }
            column++;
        }
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);
        //taskTiming[9] += (omp_get_wtime() - tstart);
    #if defined(USE_CUBLAS)
        //copying d_coordX to the device memoery for the following operations.
        cuberror = cudaMalloc ((void**)&d_coordX, gramASize * blocksize * sizeof(double));
        if( cuberror != 0 ){ printf("cudaMalloc Filed d_coordX\n"); return 0; }
        
        tstart = omp_get_wtime();
        cuberror = cudaMemcpy(d_coordX, coordX, gramASize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
        if( cuberror != 0 ){ printf("cudaMemcpy failed d_coordX ==> %d\n", cuberror); return 0; } 
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][18] += (omp_get_wtime() - tstart);   
    #endif

        if(restart == 0)
        {
            // partil result- part1:- blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:)
        #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif

        #if defined( USE_CUBLAS )
            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, currentBlockSize, 
                            &cudaAlpha, d_coordX+(blocksize*blocksize), blocksize, d_activeBlockVectorR, currentBlockSize, &cudaBeta, d_blockVectorP, blocksize); 
            
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-18: %d\n",cubstat);
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart); 
        #endif
            /*blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + blockVectorP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize+activeRSize+activePSize,:); */
        #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorP, currentBlockSize, coordX+((blocksize+currentBlockSize)*blocksize), blocksize, 1.0, blockVectorP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif
        #if defined( USE_CUBLAS )
            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, currentBlockSize, 
                            &cudaAlpha, d_coordX+((blocksize+currentBlockSize)*blocksize), blocksize, d_activeBlockVectorP, currentBlockSize, &cudaBetaOne, d_blockVectorP, blocksize);  
            
            // cuberror = cudaMemcpy(blockVectorP, d_blockVectorP, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorP: %d\n", cuberror);}
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-19: %d\n",cubstat);
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif
            /*blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + blockVectorAP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize + activeRSize+activePSize,:);*/
        #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorAR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorAP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif

        #if defined( USE_CUBLAS )
            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, currentBlockSize, 
                            &cudaAlpha, d_coordX+(blocksize*blocksize), blocksize, d_activeBlockVectorAR, currentBlockSize, &cudaBeta, d_blockVectorAP, blocksize);  
            
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-20: %d\n",cubstat);
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif

         #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorAP, currentBlockSize, coordX+((blocksize+currentBlockSize)*blocksize), blocksize, 1.0, blockVectorAP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif

        #if defined( USE_CUBLAS )
            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, currentBlockSize, 
                            &cudaAlpha, d_coordX+((blocksize+currentBlockSize)*blocksize), blocksize, d_activeBlockVectorAP, currentBlockSize, &cudaBetaOne, d_blockVectorAP, blocksize);  
           
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-21: %d\n",cubstat);
            cudaDeviceSynchronize();
             taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            // cuberror = cudaMemcpy(blockVectorAP, d_blockVectorAP, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorAP: %d\n", cuberror);}
        #endif
        }
        else
        {
            // blockVectorP =   blockVectorR(:,activeMask)*...
            //    coordX(blockSize+1:blockSize+activeRSize,:);
        #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, M, blocksize, activeRSize, 1.0, activeBlockVectorR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif

        #if defined( USE_CUBLAS )
            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, currentBlockSize, 
                            &cudaAlpha, d_coordX+(blocksize*blocksize), blocksize, d_activeBlockVectorR, currentBlockSize, &cudaBeta, d_blockVectorP, blocksize); 
            
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-22: %d\n",cubstat);
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            // cuberror = cudaMemcpy(blockVectorP, d_blockVectorP, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorP: %d\n", cuberror);} 
        #endif
            //blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:);
        #if defined(USE_BLAS)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, M, blocksize, activeRSize, 1.0, activeBlockVectorAR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorAP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        #endif

        #if defined(USE_OPENACC)
            tstart = omp_get_wtime();
            cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, currentBlockSize, 
                            &cudaAlpha, d_coordX+(blocksize*blocksize), blocksize, d_activeBlockVectorAR, currentBlockSize, &cudaBeta, d_blockVectorAP, blocksize); 
            
            if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-22: %d\n",cubstat);
            cudaDeviceSynchronize();
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            // cuberror = cudaMemcpy(blockVectorAP, d_blockVectorAP, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
            // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorAP: %d\n", cuberror);} 
        #endif
        }
    #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorX, blocksize, coordX, blocksize, 0.0, newX, blocksize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
    #endif
    #if defined( USE_CUBLAS )
        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, blocksize, 
                    &cudaAlpha, d_coordX, blocksize, d_blockVectorX, blocksize, &cudaBeta, d_newX, blocksize); 
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-23: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        // cuberror = cudaMemcpy(newX, d_newX, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed d_newX: %d\n", cuberror);} 
        //cudaDeviceSynchronize();
    #endif

     #if defined(USE_BLAS)
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorAX, blocksize, coordX, blocksize, 0.0, newX, blocksize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
    #endif
    #if defined( USE_CUBLAS )
        tstart = omp_get_wtime();
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, blocksize, M, blocksize, 
                    &cudaAlpha, d_coordX, blocksize, d_blockVectorAX, blocksize, &cudaBeta, d_temp3, blocksize); 
        
        if(cubstat != CUBLAS_STATUS_SUCCESS)
                        printf("cublasDgemm status-24: %d\n",cubstat);
        cudaDeviceSynchronize();
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        // cuberror = cudaMemcpy(temp3, d_temp3, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed d_temp3: %d\n", cuberror);} 

        //cudaDeviceSynchronize();
    #endif
    
    #if defined(USE_CPU)
        tstart = omp_get_wtime();
        mat_addition(newX, blockVectorP, blockVectorX, M, blocksize);
        taskTiming[iterationNumber - 1][3] += (omp_get_wtime() - tstart);
    #endif

    #if defined(USE_OPENACC)
        cudaDeviceSynchronize();
        tstart = omp_get_wtime();
        mat_addition_OpenACC(d_newX, d_blockVectorP, d_blockVectorX, M, blocksize);
        taskTiming[iterationNumber - 1][3] += (omp_get_wtime() - tstart);

        // cuberror = cudaMemcpy(blockVectorX, d_blockVectorX, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorX: %d\n", cuberror);} 
        //cudaDeviceSynchronize();
    #endif
    
    #if defined(USE_CPU)
        tstart = omp_get_wtime();
        mat_addition(temp3, blockVectorAP, blockVectorAX, M, blocksize);
        taskTiming[iterationNumber - 1][3] += (omp_get_wtime() - tstart);
    #endif

    #if defined(USE_OPENACC)
        tstart = omp_get_wtime();
        mat_addition_OpenACC(d_temp3, d_blockVectorAP, d_blockVectorAX, M, blocksize);
        taskTiming[iterationNumber - 1][3] += (omp_get_wtime() - tstart);
        // cudaDeviceSynchronize();
        // cuberror = cudaMemcpy(blockVectorAX, d_blockVectorAX, M * blocksize * sizeof(double), cudaMemcpyDeviceToHost);
        // if( cuberror != 0 ){ printf("cudaMemcpy failed d_blockVectorAX: %d\n", cuberror);} 
        cudaDeviceSynchronize();
    #endif
    
        //temp1Time=omp_get_wtime();
        delete []eigen_value;
        // delete []work;
        //delete []gramA;
        //delete []gramB;
        delete []coordX;
    
    #if defined( USE_CUBLAS )
        cudaFree(d_coordX); 
        cudaDeviceSynchronize();
    #endif
        
        prevCurrentBlockSize = currentBlockSize;
        
        loopTime[iterationNumber - 1] = omp_get_wtime() - loop_start_time;
        for(i = 0 ; i < blocksize ; i++)
        {
            saveLamda[i][iterationNumber - 1] = lambda[i * blocksize + i];
        }

        /*printf("%10s %.6lf sec.\n", "SETZERO", taskTiming[0]);
        printf("%10s %.6lf sec.\n", "XY", taskTiming[1]);
        printf("%10s %.6lf sec.\n", "XTY", taskTiming[2]);
        printf("%10s %.6lf sec.\n", "ADD", taskTiming[3]);
        printf("%10s %.6lf sec.\n", "SUB", taskTiming[4]);
        printf("%10s %.6lf sec.\n", "MULT", taskTiming[5]);
        printf("%10s %.6lf sec.\n", "SPMM", taskTiming[6]);
        printf("%10s %.6lf sec.\n", "GET", taskTiming[7]);
        printf("%10s %.6lf sec.\n", "UPDATE", taskTiming[8]);
        printf("%10s %.6lf sec.\n", "EIGEN", taskTiming[9]);
        printf("%10s %.6lf sec.\n", "DLACPY", taskTiming[10]);*/

    } //loop ends

#if defined(USE_CUBLAS)  
    cudaFree(d_blockVectorX);
    cudaFree(d_blockVectorAX);
    cudaFree(d_blockVectorR);
    cudaFree(d_blockVectorP);
    cudaFree(d_blockVectorAP);
    cudaFree(d_newX);

    cudaFree(d_activeBlockVectorR);
    cudaFree(d_activeBlockVectorAR);
    cudaFree(d_activeBlockVectorP);
    cudaFree(d_activeBlockVectorAP);
    cudaFree(d_temp3);

    cudaFree(d_temp2);
    cudaFree(d_lambda);
    cudaFree(d_gramRBR);
    cudaFree(d_gramPBP);
    cudaFree(d_gramXAR);
    cudaFree(d_gramRAR);

    cudaFree(d_gramXAP);
    cudaFree(d_gramPAP);
    cudaFree(d_gramRAP);
    cudaFree(d_gramXBP);
    cudaFree(d_gramRBP);

    cublasDestroy(handle); 
#endif 

    //iteraton_time = omp_get_wtime() - iteraton_start_time;
    
    // cout << "Total iterations: " << iterationNumber -1 << endl;
    //cout << "Total Execution time: " << iteraton_time << " sec." << endl;

    //printf("LoopTime: \n");
    double totalSum = 0;
    for(j = 3 ; j < maxIterations ; j++)
    {
        totalSum += loopTime[j];
        printf("%.4lf,", loopTime[j]);
        //if(j != maxIterations - 1)
        //    printf(",");
    }
    printf("%.4lf", totalSum/(maxIterations-3));
    printf("\n");

    //printing eigen values
    for(i = 0 ; i < blocksize ; i++)
    {
        for(j = 0 ; j < maxIterations ; j++)
        {
            printf("%.4lf", saveLamda[i][j]);
            if(j != maxIterations - 1)
                printf(",");
        }
        printf("\n");
    }
    double avgTotal = 0.0;
    for(j = 0 ; j < numTasks ; j++)
    {
        totalSum = 0.0;
        for(i = 3 ; i < maxIterations ; i++)
        {
            totalSum += taskTiming[i][j];
        }
        avgTotal += totalSum/(maxIterations - 3);
        printf("%s,%.6lf\n", function_name[j].c_str(), totalSum/(maxIterations - 3));
    }
    printf("Kernel Total,%.6lf\n", avgTotal);

    return 0;
}
