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

#include <mkl.h>
#include <omp.h>
#include "lib_csr_openacc.h"

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
    stringstream s1(argv[2]);
    s1 >> block_width;
    cout << "Block Size: " << blocksize << " Block Width: " << block_width << endl;

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
    char *filename = argv[3] ; 
    wblk = block_width; 
    // read_custom(filename,xrem);
    // printf("Finish Reading CUS file\n");

    // *--------- Reading CSR from binary file ---------* //
    ifstream file ("Nm7_CSR.dat", ios::in|ios::binary);
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
    cout << "ia[0]: " << ia[0] << " ia[last]: " << ia[numrows] << endl;
    cout << "ja[0]: " << ja[0] << " ja[last]: " << ja[nnonzero-1] << endl;
    cout << "acsr[0]: " << acsr[0] << " acsr[last]: " << acsr[nnonzero-1] << endl;
    printf("numrows: %d numcols: %d nnonzero: %d\n", numrows, numcols, nnonzero);
    // *--------- Reading from txt file finished ---------* //
    
    M = numrows;
    N = numcols;
    
    #pragma omp parallel
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }

    //* ------ deleting CSC storage memory ------ *//
    // delete []colptrs;
    // delete []irem;
    // delete []xrem;

    

    //timing variables
    int numTasks = 18;
    vector<string> function_name{"LOOPS", "X*Y", "Xt*Y", "ADD", "SUB", "MULT", "SPMM", "GET", "UPDATE", "dsygv", "DLACPY", "INVERSE", "TRANSPOSE", "mat_copy", "dpotrf", "memset", "SUMSQRT", "diag"};
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

    // cout << "acsr[0]: " << acsr[0] << "acsr[last]: " << acsr[nnonzero-1] << endl;
    
    //printf("mkl_dcsrcsc: %lf sec.\n", omp_get_wtime() - tstart);

    // *------ deleting CSC storage memory ------* //
    // delete []colptrs;
    // delete []irem;
    // delete []xrem;

    // *------ Writing CSR format data to files ------* //
    // ofstream acsrfile, jafile, iafile;
    // acsrfile.open ("inline1_acsr.txt");
    // jafile.open ("inline1_ja.txt");
    // iafile.open ("inline1_ia.txt");
    
    // acsrfile << nnonzero << endl;
    // jafile << nnonzero << endl;
    // iafile << numrows + 1 << endl;

    // acsrfile << std::fixed << std::setprecision(3);
    // for (i = 0 ; i < nnonzero; i++)
    // {
    //     acsrfile  << acsr[i] << endl;
    //     jafile  << ja[i] << endl;
    // }
    // for (i = 0 ; i <= numrows; i++)
    // {
    //     iafile  << ia[i] << endl;
    // }
    // acsrfile.close();
    // jafile.close();
    // iafile.close();
    // printf("**--- Writing into file finished --**\n");
    // exit(1);
    // *------ Writing CSR format data to files Finished ------* //

    // *--------- Reading from txt file ---------* //
    // std::ifstream acsrfile("inline1_acsr.txt");
    // acsrfile >> nnonzero;
    // acsr = (double *) malloc(nnonzero * sizeof(double)); //xrem
    // for(i = 0; i < nnonzero ; i++){
    //     acsrfile >> acsr[i];
    //     if(i == (nnonzero - 1))
    //         cout << "acsr[last]: " << acsr[i] << endl;
    // }
    // acsrfile.close();

    // std::ifstream jafile("inline1_ja.txt");
    // jafile >> nnonzero;
    // ja = (int *) malloc(nnonzero * sizeof(int)); //irem
    // for(i = 0; i < nnonzero ; i++){
    //     jafile >> ja[i];
    //     if(i == (nnonzero - 1))
    //         cout << "ja[last]: " << ja[i] << endl;
    // }
    // jafile.close();
    
    // std::ifstream iafile("inline1_ia.txt");
    // iafile >> numrows;
    // numrows = numrows - 1;
    // numcols = numrows;
    // ia = (int *) malloc((numrows + 1) * sizeof(int)); //colsptr
    // for(i = 0; i < (numrows  + 1) ; i++){
    //     iafile >> ia[i];
    //     if(i == numrows)
    //         cout << "ia[last]: " << ia[i] << endl;
    // }
    // iafile.close();
    // printf("numrows: %d numcols: %d nnonzero: %d\n", numrows, numcols, nnonzero);
    // *--------- Reading from txt file finished ---------* //

    // *--------- Writing CSR format in binary file ---------* //
    // std::ofstream mtxfile;
    // mtxfile.open("Nm7_CSR.dat", ios::binary | ios::out);
    // mtxfile.write(reinterpret_cast<char*>(&numrows), sizeof(int));
    // mtxfile.write(reinterpret_cast<char*>(&numrows), sizeof(int));
    // mtxfile.write(reinterpret_cast<char*>(&nnonzero), sizeof(int));
    // mtxfile.write(reinterpret_cast<char*>(ia), (numrows + 1) * sizeof(int));
    // mtxfile.write(reinterpret_cast<char*>(ja), nnonzero * sizeof(int));
    // mtxfile.write(reinterpret_cast<char*>(acsr), nnonzero * sizeof(double));

    // i = 0;
    //     while(i <= numrows)
    //     {
    //         j = ia[i++];
    //         mtxfile.write(reinterpret_cast<char*>(&j), sizeof(j));
    //     }
    //     cout << "finished writing ia"<<endl;
    //     i = 0;
    //     while(i < nnonzero)
    //     {
    //         j = ja[i++];
    //         mtxfile.write(reinterpret_cast<char*>(&j), sizeof(j));
    //     }
    //     cout << "finished writing ja"<<endl;
    //     i = 0;
    //     double d;
    //     while(i < nnonzero)
    //     {
    //         d = acsr[i++];
    //         mtxfile.write(reinterpret_cast<char*>(&d), sizeof(d));
    //     }  
    //     cout << "finished reading acsr"<<endl;
    // mtxfile.close();
    // printf("numrows: %d numcols: %d nnonzero: %d\n", numrows, numcols, nnonzero);
    // exit(1);
    // *--------- Writing CSR format in binary file finished ---------* //
    
    // M = numrows;
    // N = numcols;
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

    double *gramA, *gramB;
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
    
    std::memset(zeros_B_CB, 0.0, sizeof(zeros_B_CB));
    std::memset(zeros_CB_B, 0.0, sizeof(zeros_CB_B));

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

    //* -- changing LAPACKE_dpotrf to dpotrf_
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR , 'U' , blocksize , gramXBX , blocksize);

    // double *trans_gramXBX = new double[blocksize * blocksize]();
    // transpose(gramXBX, trans_gramXBX, blocksize, blocksize);
    // dpotrf_( &uplo, &blocksize, trans_gramXBX, &blocksize, &info );
    // transpose(trans_gramXBX, gramXBX, blocksize, blocksize);
    // delete []trans_gramXBX;

    if(info != 0)
    {
        cout << "dpotrf: chol error!" << endl;
        //exit(1);
    }
    
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
    
    // printf("**---------------- Printing blockVectorAX ----------------**\n");
    // print_mat(blockVectorAX, 4, blocksize);

    //gramXAX = full(blockVectorX'*blockVectorAX);
    double *gramXAX=new double[blocksize*blocksize]();
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,blocksize,blocksize,M,1.0,blockVectorX,blocksize,blockVectorAX,blocksize,0.0,gramXAX,blocksize);
    //_XTY(blockVectorX, blockVectorAX, gramXAX, M, blocksize, blocksize, block_width);
    
    //gramXAX = (gramXAX + gramXAX')*0.5;
    double *transGramXAX = new double[blocksize*blocksize]();
    transpose(gramXAX, transGramXAX, blocksize, blocksize);
    
    make_identity_mat(identity_BB,blocksize, blocksize);
    make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize); //--> used in loop
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blocksize, blocksize, blocksize, 0.5, transGramXAX, blocksize, identity_BB, blocksize, 0.5, gramXAX, blocksize);
    free(transGramXAX);
    
    
    
    
    

    // *---------------- Changing dsygv_ to LAPACKE_dsygv ----------------* //
    double *tempLambda = new double[blocksize]();
    info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, blocksize, gramXAX, blocksize, identity_BB, blocksize, tempLambda);
    
    // double *temp_GramXAX = new double[blocksize * blocksize]();
    // transpose(gramXAX, temp_GramXAX, blocksize, blocksize);
    // lwork = -1;
    // dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, &work_query, &lwork, &info);
    // if(info != 0)
    //   cout << "Error in dummy call" << endl;
    // lwork = (int) work_query;
    // work = new double[lwork]();
    // dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, work, &lwork, &info);
    // transpose(temp_GramXAX, gramXAX, blocksize, blocksize);
    // free(temp_GramXAX);
    // free(work);
    // if(info != 0)
    //     printf( "The algorithm failed to compute eigenvalues.\n" );
    
    //[coordX,gramXAX]=eig(gramXAX,eye(blockSize));
    //lambda=diag(gramXAX);
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
    

    int gramASize = -1;
    int *activeMask = (int *) malloc(blocksize * sizeof(int));

    #pragma omp parallel for
    for(i = 0 ; i < blocksize ; i++)
        activeMask[i] = 1;

    iteraton_start_time = omp_get_wtime();

    int activeRSize = 0, activePSize = 0, explicitGramFlag = 0, restart = 0;
    
    //loop starts here
    for(iterationNumber = 1 ; iterationNumber <= maxIterations ; iterationNumber++)
    {
        // for(i = 0 ; i < numTaks ; i++)
        //     taskTiming[i] = 0;

        //cout << "\niterationNumber: " << iterationNumber << endl;
        loop_start_time = omp_get_wtime();
        printf("**---------------- iterationNumber : %ld ----------------**\n", iterationNumber);
        //if 12 nested if
        //blockVectorR = blockVectorAX - blockVectorX*spdiags(lambda,0,blockSize,blockSize);
        
        // printf("iterationNumber : %ld blockVectorX ==>\n", iterationNumber);
        // print_mat(blockVectorX, 2, blocksize);
        // printf("\n");

        // printf("iterationNumber : %ld lambda ==>\n", iterationNumber);
        // print_mat(lambda, 2, blocksize);
        // printf("\n");

        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorX, blocksize, lambda, blocksize, 0.0, blockVectorR, blocksize); //XY code 1
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);

        // printf("iterationNumber : %ld blockVectorR ==>\n", iterationNumber);
        // print_mat(blockVectorR, 2, blocksize);
        // printf("\n");

        tstart = omp_get_wtime();
        mat_sub(blockVectorAX, blockVectorR, blockVectorR , M, blocksize); //SUB : 4
        taskTiming[iterationNumber - 1][4] += (omp_get_wtime() - tstart); 
        
        // printf("iterationNumber : %ld blockVectorR ==>\n", iterationNumber);
        // print_mat(blockVectorR, 2, blocksize);
        // printf("\n");

        //residualNorms=full(sqrt(sum(conj(blockVectorR).*blockVectorR)')); 
        tstart = omp_get_wtime();
        #pragma omp parallel for default(shared)
        for(i = 0 ; i < blocksize ; i++)
            residualNorms[i] = 0.0;
        taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        mat_mult(blockVectorR, blockVectorR, newX, M, blocksize); //MULT : 5
        taskTiming[iterationNumber - 1][5] += (omp_get_wtime() - tstart);

        // printf("iterationNumber : %ld newX ==>\n", iterationNumber);
        // print_mat(newX, 2, blocksize);
        // printf("\n");
        
        tstart = omp_get_wtime();
        sum_sqrt(newX, residualNorms, M, blocksize);
        taskTiming[iterationNumber - 1][16] += (omp_get_wtime() - tstart);
        
        // printf("iterationNumber : %ld residualNorms ==>\n", iterationNumber);
        // print_mat(residualNorms, 1, blocksize);
        // printf("\n");
        //residualNormsHistory(1:blockSize,iterationNumber)=residualNorms;
        //activeMask = full(residualNorms > residualTolerance) & activeMask;

        tstart = omp_get_wtime();
        update_activeMask(activeMask, residualNorms, residualTolerance, blocksize);
        taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart); //UPDATE : 8

        // printf("iterationNumber : %ld blockVectorR ==>\n", iterationNumber);
        // print_mat(blockVectorR, 2, blocksize);
        // printf("\n");

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

        tstart = omp_get_wtime();
        getActiveBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize); //GET: 7
        taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);

        // printf("iterationNumber : %ld activeBlockVectorR ==>\n", iterationNumber);
        // print_mat(activeBlockVectorR, 2, currentBlockSize);
        // printf("\n");

        //blockVectorX'*blockVectorR(:,activeMask)  -> temp2 is the result
        tstart = omp_get_wtime();
        std::memset(temp2, 0.0, sizeof(temp2));
        taskTiming[iterationNumber - 1][15] += (omp_get_wtime() - tstart);

        // printf("iterationNumber : %ld blockVectorX ==>\n", iterationNumber);
        // print_mat(blockVectorX, 2, blocksize);
        // printf("\n");

        // printf("iterationNumber : %ld activeBlockVectorR ==>\n", iterationNumber);
        // print_mat(activeBlockVectorR, 2, currentBlockSize);
        // printf("\n");
        
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorX, blocksize, activeBlockVectorR, currentBlockSize,0.0, temp2, currentBlockSize); //XTY : 2
        //_XTY(blockVectorX, activeBlockVectorR, temp2, M, blocksize, currentBlockSize, block_width);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

        // printf("iterationNumber : %ld temp2 ==>\n", iterationNumber);
        // print_mat(temp2, 2, currentBlockSize);
        // printf("\n");

        // printf("iterationNumber : %ld blockVectorX ==>\n", iterationNumber);
        // print_mat(blockVectorX, 2, blocksize);
        // printf("\n");
    
        //temp3 = blockVectorX * temp2
        

        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, blocksize, 1.0, blockVectorX, blocksize, temp2, currentBlockSize, 0.0, temp3, currentBlockSize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        mat_sub(activeBlockVectorR, temp3, activeBlockVectorR, M, currentBlockSize);
        taskTiming[iterationNumber - 1][4] += (omp_get_wtime() - tstart);

        // tstart = omp_get_wtime();
        // updateBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize); //UPDATE: 8
        // taskTiming[8] += (omp_get_wtime() - tstart);
    
        //------- if 18 ------
        //gramRBR=blockVectorR(:,activeMask)'*blockVectorR(:,activeMask);  //blockVectorR(:,activeMask) ->activeBlockVectorR

        

        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorR, currentBlockSize, activeBlockVectorR, currentBlockSize, 0.0,gramRBR, currentBlockSize);
        //_XTY(activeBlockVectorR, activeBlockVectorR, gramRBR, M, currentBlockSize, currentBlockSize, block_width);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
        
        // * --------- Changing dpotrf_ to LAPACKE_dpotrf --------- * //
        
        //[gramRBR,cholFlag]=chol(gramRBR);

        tstart = omp_get_wtime();
        info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR , 'U' , currentBlockSize , gramRBR , currentBlockSize);
        taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);
        
        // tstart = omp_get_wtime();
        // transpose(gramRBR, trans_gramRBR, currentBlockSize, currentBlockSize);
        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);
        
        // tstart = omp_get_wtime();
        // dpotrf_( &uplo, &currentBlockSize, trans_gramRBR, &currentBlockSize, &info );
        // taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);
        
        // tstart = omp_get_wtime();
        // transpose(trans_gramRBR, gramRBR, currentBlockSize, currentBlockSize);
        // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

        if(info != 0)
        {
            cout << "dportf_ error 2!!" << endl;
            break;
        }
        

        tstart = omp_get_wtime();
        #pragma acc parallel loop private(j)
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

            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorR, currentBlockSize, gramRBR, currentBlockSize, 0.0, temp3, currentBlockSize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);

            tstart = omp_get_wtime();
            custom_dlacpy(temp3, activeBlockVectorR, M, currentBlockSize); //DLACPY: 11
            taskTiming[iterationNumber - 1][10] += (omp_get_wtime() - tstart);
            
            tstart = omp_get_wtime();
            updateBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize);
            taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);
        } //end if

        

        tstart = omp_get_wtime();
        #pragma acc parallel loop private(j)
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

        // printf("**---------------- Printing activeBlockVectorAR ----------------**\n");
        // print_mat(activeBlockVectorAR, 4, blocksize);
        // printf("\n");

        // tstart = omp_get_wtime();
        // updateBlockVector(activeBlockVectorAR, activeMask, blockVectorAR, M, blocksize, currentBlockSize);
        // taskTiming[8] += (omp_get_wtime() - tstart);

        if(iterationNumber > 1)
        {
            //if 20 first nested if
            // gramPBP=blockVectorP(:,activeMask)'*blockVectorP(:,activeMask);
            tstart = omp_get_wtime();
            getActiveBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);
            taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);

            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorP, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramPBP, currentBlockSize);
            //_XTY(activeBlockVectorP, activeBlockVectorP, gramPBP, M, currentBlockSize, currentBlockSize, block_width);
            taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
            
            // * --------- Changing dpotrf_ to LAPACKE_dpotrf --------- * //
            tstart = omp_get_wtime();
            info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR , 'U' , currentBlockSize , gramPBP , currentBlockSize);
            taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);
            
            // tstart = omp_get_wtime();
            // transpose(gramPBP, trans_gramPBP, currentBlockSize, currentBlockSize);
            // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);            
            
            // tstart = omp_get_wtime();
            // dpotrf_( &uplo, &currentBlockSize, trans_gramPBP, &currentBlockSize, &info );
            // taskTiming[iterationNumber - 1][14] += (omp_get_wtime() - tstart);

            // tstart = omp_get_wtime();
            // transpose(trans_gramPBP, gramPBP, currentBlockSize, currentBlockSize);
            // taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

            if(info != 0)
            {
                cout << "dportf_ error 3" << endl;
                break;
            }
            
            //making the lower part of gramPBP zero
            tstart = omp_get_wtime();
            #pragma acc parallel loop private(j)
            for(i = 0 ; i < currentBlockSize ; i++)
            {
                for(j = 0 ; j < i ; j++)
                {
                    gramPBP[i * currentBlockSize + j] = 0.0;
                }
            }
            // printf("**---------------- Printing gramPBP ----------------**\n");
            // print_mat(gramPBP, 4, currentBlockSize);
            // printf("\n");

            taskTiming[iterationNumber - 1][0] += (omp_get_wtime() - tstart);

            if(info == 0)
            {
                //if 20 first nested if 2
                // blockVectorP(:,activeMask) = blockVectorP(:,activeMask)/gramPBP;
                tstart = omp_get_wtime();
                inverse(gramPBP, currentBlockSize, currentBlockSize);
                taskTiming[iterationNumber - 1][11] += (omp_get_wtime() - tstart);
              
                tstart = omp_get_wtime();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorP, currentBlockSize, gramPBP, currentBlockSize, 0.0, temp3, currentBlockSize);
                taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
                
                tstart = omp_get_wtime();
                custom_dlacpy(temp3, activeBlockVectorP, M, currentBlockSize);
                taskTiming[iterationNumber -1][10] += (omp_get_wtime() - tstart);
             
                tstart = omp_get_wtime();
                updateBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);

             
                //blockVectorAP(:,activeMask) = blockVectorAP(:,activeMask)/gramPBP;
                tstart = omp_get_wtime();
                getActiveBlockVector(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][7] += (omp_get_wtime() - tstart);
                
                tstart = omp_get_wtime();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorAP, currentBlockSize, gramPBP, currentBlockSize, 0.0, temp3, currentBlockSize);
                taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
                
                tstart = omp_get_wtime();
                custom_dlacpy(temp3, activeBlockVectorAP, M, currentBlockSize);
                taskTiming[iterationNumber - 1][10] += (omp_get_wtime() - tstart);
                
                tstart = omp_get_wtime();
                updateBlockVector(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize);
                taskTiming[iterationNumber - 1][8] += (omp_get_wtime() - tstart);

                // printf("**---------------- Printing activeBlockVectorP ----------------**\n");
                // print_mat(activeBlockVectorP, 4, currentBlockSize);
                // printf("\n");
                // printf("**---------------- Printing activeBlockVectorAP ----------------**\n");
                // print_mat(activeBlockVectorAP, 4, currentBlockSize);
                // printf("\n");

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
 
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorAX, blocksize, activeBlockVectorR, currentBlockSize, 0.0, gramXAR, currentBlockSize);
        //_XTY(blockVectorAX, activeBlockVectorR, gramXAR, M, blocksize, currentBlockSize, block_width);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
     
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAR, currentBlockSize, activeBlockVectorR, currentBlockSize, 0.0, gramRAR, currentBlockSize);
        //_XTY(activeBlockVectorAR, activeBlockVectorR, gramRAR, M, currentBlockSize, currentBlockSize, block_width);
        taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize);
        taskTiming[iterationNumber - 1][12] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, transGramRAR, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramRAR, currentBlockSize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);

     
        //--- cond_try for loop -----
        for(int cond_try = 1 ; cond_try <=2 ; cond_try++)
        {
            if(restart == 0) //---- if 24 ----
            {
                if(restart == 0)
                {
                    //cout<<"if 24"<<endl;
                    //gramXAP=full(blockVectorAX'*blockVectorP(:,activeMask));

                    //activeBlockVectorP= new double[M*currentBlockSize]();
                    //getActiveBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);
                    
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorAX, blocksize, activeBlockVectorP, currentBlockSize, 0.0, gramXAP, currentBlockSize);
                    //_XTY(blockVectorAX, activeBlockVectorP, gramXAP, M, blocksize, currentBlockSize, block_width);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                 
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAR, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramRAP, currentBlockSize);
                    //_XTY(activeBlockVectorAR, activeBlockVectorP, gramRAP, M, currentBlockSize, currentBlockSize, block_width);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                    
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAP, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramPAP, currentBlockSize);
                    //_XTY(activeBlockVectorAP, activeBlockVectorP, gramPAP, M, currentBlockSize, currentBlockSize, block_width);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                    
                    // printf("**---------------- Printing gramPAP 1 ----------------**\n");
                    // print_mat(gramPAP, currentBlockSize, currentBlockSize);
                    // printf("\n");
                    //gramPAP=(gramPAP'+gramPAP)*0.5;

                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, gramPAP, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramPAP, currentBlockSize);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

                    // printf("**---------------- Printing gramPAP 2 ----------------**\n");
                    // print_mat(gramPAP, currentBlockSize, currentBlockSize);
                    // printf("\n");
                 
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

                        gramASize = blocksize + currentBlockSize + currentBlockSize;

                        gramA = new double[gramASize * gramASize]();
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
                    } //end else
                    //double *gramXBP = new double[blocksize*currentBlockSize]();
                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorX, blocksize, activeBlockVectorP, currentBlockSize, 0.0, gramXBP, currentBlockSize);
                    //_XTY(blockVectorX, activeBlockVectorP, gramXBP, M, blocksize, currentBlockSize, block_width);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);

                    tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorR, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramRBP, currentBlockSize);
                    //_XTY(activeBlockVectorR, activeBlockVectorP, gramRBP, M, currentBlockSize, currentBlockSize, block_width);
                    taskTiming[iterationNumber - 1][2] += (omp_get_wtime() - tstart);
                 
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

                        gramB = new double[gramASize * gramASize];
                        
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
                    gramASize = blocksize + activeRSize;
                    gramA = new double[gramASize * gramASize]();
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
                    
                    gramB = new double[gramASize * gramASize]();
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

        // printf("**---------------- Printing gramB ----------------**\n");
        // print_mat(gramB, 4, gramASize);
        // printf("\n");
        // if(iterationNumber == 2)
        // {
        //     ofstream gafile, gbfile;
        //     gafile.open ("ga_v1_itr_2.txt");
        //     gbfile.open ("gb_v1_itr_2.txt");
        //     for(i = 0 ; i < gramASize ; i++)
        //     {
        //         for(j = 0 ; j < gramASize ; j++)
        //         {
        //             gafile << gramA[i * gramASize + j] << " ";
        //             gbfile << gramB[i * gramASize + j] << " ";
        //         }
        //         gafile << endl;
        //         gbfile << endl;
        //     }
        //     gafile.close();
        //     gbfile.close();
        // }

        // *---------------- Changing dsygv_ to LAPACKE_dsygv ----------------* //
        eigen_value = new double[gramASize]();

        tstart = omp_get_wtime();
        info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, gramASize, gramA, gramASize, gramB, gramASize, eigen_value);
        taskTiming[iterationNumber - 1][9] += (omp_get_wtime() - tstart);
        
        // printf("**---------------- Printing eigen_value ----------------**\n");
        // print_mat(eigen_value, 1, gramASize);
        // printf("\n");

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
      
        if(info != 0)
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

        if(restart == 0)
        {
            // partil result- part1:- blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:)
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            
            /*blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + blockVectorP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize+activeRSize+activePSize,:); */
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorP, currentBlockSize, coordX+((blocksize+currentBlockSize)*blocksize), blocksize, 1.0, blockVectorP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
         
            /*blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + blockVectorAP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize + activeRSize+activePSize,:);*/

            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorAR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorAP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
            
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorAP, currentBlockSize, coordX+((blocksize+currentBlockSize)*blocksize), blocksize, 1.0, blockVectorAP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        }
        else
        {
            //cout<<"if 26 else"<<endl;
            // blockVectorP =   blockVectorR(:,activeMask)*...
            //    coordX(blockSize+1:blockSize+activeRSize,:);

            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, M, blocksize, activeRSize, 1.0, activeBlockVectorR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
         
            //blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:);
        
            tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, M, blocksize, activeRSize, 1.0, activeBlockVectorAR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorAP, blocksize);
            taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);
        }

        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorX, blocksize, coordX, blocksize, 0.0, newX, blocksize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        mat_addition(newX, blockVectorP, blockVectorX, M, blocksize);
        taskTiming[iterationNumber - 1][3] += (omp_get_wtime() - tstart);
        
        tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorAX, blocksize, coordX, blocksize, 0.0, newX, blocksize);
        taskTiming[iterationNumber - 1][1] += (omp_get_wtime() - tstart);

        tstart = omp_get_wtime();
        mat_addition(newX, blockVectorAP, blockVectorAX, M, blocksize);
        taskTiming[iterationNumber - 1][3] += (omp_get_wtime() - tstart);
        
        //temp1Time=omp_get_wtime();
        delete []eigen_value;
        // delete []work;
        delete []gramA;
        delete []gramB;
        delete []coordX;
        
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
        avgTotal += totalSum/(maxIterations - 2);
        printf("%s,%.6lf\n", function_name[j].c_str(), totalSum/(maxIterations - 2));
    }
    printf("Kernel Total,%.6lf\n", avgTotal);

    return 0;
}
