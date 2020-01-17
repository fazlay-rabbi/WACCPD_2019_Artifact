#ifndef LIB_CSR_H
#define LIB_CSR_H
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
using namespace std;

// #include <mkl.h>
#include <openacc.h>
#include <omp.h>
#include <cblas.h>
#include <lapacke.h>

#include "lib_csr_openacc.h"

long position = 0;
int *colptrs, *irem;
int  *ia , *ja;

int numcols, numrows, nnonzero, nthrds = 1;
int nrows, ncols, nnz;
int wblk, nrowblks, ncolblks;

template<typename T>
struct block
{
    int nnz;
    int roffset, coffset;
    unsigned short int *rloc, *cloc;
    T *val;
};

template<typename T>
void read_custom(char* filename, T *&xrem)
{
    int i,j;
    ifstream file (filename, ios::in|ios::binary);
    if (file.is_open())
    {
        int a = 0, c=0;
        long int b=0;
        //float d=0;
        float d=0;
        file.read ((char*)&numrows,sizeof(numrows));
        cout<<"row: "<<numrows<<endl;
        file.read(reinterpret_cast<char*>(&numcols), sizeof(numcols));
        cout<<"colum: "<<numcols<<endl;

        file.read(reinterpret_cast<char*>(&nnonzero), sizeof(nnonzero));
        cout<<"non zero: "<<nnonzero<<endl;

        colptrs=new int[numcols+1];
        irem=new int[nnonzero];
        xrem=new T[nnonzero];
        cout<<"Memory allocaiton finished"<<endl;
        position=0;
        while(!file.eof() && position<=numcols)
        {
            file.read(reinterpret_cast<char*>(&a), sizeof(a)); //irem(j)
            colptrs[position++] = a-1;
        }
        cout<<"finished reading colptrs"<<endl;
        position=0;
        while(!file.eof() && position<nnonzero)
        {
            //file.read ((char*)&a,sizeof(double));
            file.read(reinterpret_cast<char*>(&a), sizeof(a)); //irem(j)
            irem[position++] = a-1;
            //position++;
        }

        position=0;
        while(!file.eof() && position<nnonzero)
        {
            //file.read ((char*)&a,sizeof(double));
            file.read(reinterpret_cast<char*>(&d), sizeof(d)); //irem(j)
            xrem[position++]=d;
            //if(file.eof())
            //cout<<"EOF found.. position: "<<position<<endl;
        }  
    }
}

template struct block<double>;
template struct block<float>;

template void read_custom<double>(char* filename, double *&xrem);

void _XTY(double *X, double *Y, double *result ,int M, int N, int P, int blocksize)
{
    /*******************
    Input: X[M*N], Y[M*P]
    Output: result[N*P]
    ********************/

    int i, j, k, blksz, tid, nthreads;
    double sum,tstart,tend;
    
    #pragma omp parallel shared(nthreads)
    nthreads=omp_get_num_threads();
    
    double *buf = new double[nthreads * N * P]();


    //--- task based implementation
    #pragma omp parallel num_threads(nthreads)\
    shared(nthreads)
    {
        #pragma omp single
        {
            for(k=0;k<M;k=k+blocksize)
            {
                //tid=omp_get_thread_num();
                blksz=blocksize;
                if(k+blksz>M)
                    blksz=M-k;
                #pragma omp task firstprivate(k, blksz, tid) shared(X, Y, buf, M, N, P)
                {
                    tid=omp_get_thread_num();
                    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,N,P,blksz,1.0,X+(k * N),N,Y+(k * P),P,1.0,buf+(tid * N * P),P);
                }
            }
        }
    }
    #pragma omp taskwait

  //--------task based summation of arrays

  #pragma omp parallel num_threads(nthreads)\
  shared(nthreads, result)
  {
    #pragma omp single
    {
      for(i=0;i<N;i++)
      {
        #pragma omp task firstprivate(sum, i) private(k) shared(nthreads, result, N, P)
        {
          for(k=0; k<P; k++)
          {
            sum=0.0;
            for(j=0;j<nthreads;j++) //for each thread access corresponding N*N matrix
            {
              sum+=buf[j*N*P+i*P+k];
            }
            result[i*P+k]=sum;
          }
        }
      }
    }
  }

  #pragma omp taskwait

  delete[] buf;
}

void transpose(double *src, double *dst, const int N, const int M)
{
    int i, j;
    // #pragma acc parallel loop private(j)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < N ; j++)
        {
            dst[j * M + i] = src[i * N + j];
        }
    }
}

// void inverse(double *arr, int m, int n)
// {   
//     /**************
//     input: arr[m*n] in row major format.
//     **************/

//     int lda_t = m;
//     int lda = n;
//     int info;
//     int lwork = -1;
//     double* work = NULL;
//     double work_query;
//     int *ipiv = new int[n+1]();

//     double *arr_t = new double[m*n]();
//     transpose(arr, arr_t, n, m);
//     dgetrf_( &n, &m, arr_t, &lda_t, ipiv, &info );
//     if(info < 0)
//     {
//        cout << "dgetrf_: Transpose error!!" << endl;
//        exit(1);
//     }
//    //transpose(arr_t, arr, m, n);
//    //LAPACKE_dgetri(LAPACK_ROW_MAJOR, n,arr,n,ipiv);

//    /* Query optimal working array(s) size */
//    dgetri_( &m, arr_t, &lda_t, ipiv, &work_query, &lwork, &info );
//    if(info<0)
//    {
//        cout<<"dgetri_ 1: Transpose error!!"<<endl;
//        //exit(1);
//    }
//    lwork = (int)work_query;
//    //cout<<"lwork: "<<lwork<<endl;
//    work = new double[lwork]();
//    dgetri_( &m, arr_t, &lda_t, ipiv, work, &lwork, &info );
//    if(info<0)
//    {
//        cout<<"dgetri_ 2: Transpose error!!"<<endl;
//        //exit(1);
//    }
//    transpose(arr_t, arr, m, n);
//    delete []arr_t;
//    delete []ipiv;
// }

void inverse(double* A, int N, int m)
{
    int *IPIV = new int[N + 1];
    //int LWORK = N*N;
    //double *WORK = new double[LWORK];
    int INFO;

    INFO = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A, N, IPIV);
    if(INFO<0)
    {
        cout << "LAPACKE_dgetrf: error!!"<<endl;
        //exit(1);
    }
    //LAPACKE_dgetrf (int matrix_layout , lapack_int m , lapack_int n , double * a , lapack_int lda , lapack_int * ipiv );
    INFO = LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, A, N, IPIV);
    //lapack_int LAPACKE_dgetri (int matrix_layout , lapack_int n , double * a , lapack_int lda , const lapack_int * ipiv );
    if(INFO<0)
    {
        cout << "LAPACKE_dgetri: error!!"<<endl;
        //exit(1);
    }

    delete IPIV;
    //delete WORK;
}



void print_mat(double *arr, const int row, const int col) 
{
    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    //cout.precision(15);
    for(int i = 0 ; i < row ; i++)
    {
        for(int j = 0 ; j < col ; j++)
        {
          cout << arr[i * col + j] << " ";
        }
        cout << endl;
    }
}

void make_identity_mat(double *arr, const int row, const int col)
{
    int i, j;
    //#pragma acc parallel loop private(j)
    for(i = 0 ; i < row ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            if(i == j)
                arr[i * row + j] = 1.00;
            else
                arr[i * row + j] = 0.00;
        }
    }
}

void diag(double *src, double *dst, const int size)
{
    int i, j;
    #pragma acc parallel loop private(j)
    for(i = 0; i < size ; i++)
    {
        for(j = 0 ; j < size ; j++)
        {
            if(i == j)
            {
                dst[i * size + j] = src[i];
            }
            else
                dst[i * size + j] = 0.0;
        }
    }
}

void mat_sub(double *src1, double *src2, double *dst, const int row, const int col)
{
    int i, j;
    #pragma acc parallel loop
    for(i = 0; i < row ; i++)
    {
        #pragma acc loop
        for(j = 0 ; j < col ; j++)
        {
            dst[i * col + j] = src1[i * col + j] - src2[i * col + j];
        }
    }
}

void mat_addition(double *src1, double *src2, double *dst, const int row, const int col)
{
    int i, j;
    #pragma acc parallel loop 
    for(i = 0 ; i < row ; i++)
    {
        #pragma acc loop
        for(j = 0 ; j < col ; j++)
        {
            dst[i * col + j] = src1[i * col + j] + src2[i * col + j];
        }
    }
}

void mat_mult(double *src1, double *src2, double *dst, const int row, const int col)
{
    int i, j;
    #pragma acc parallel loop 
    for(i = 0; i < row ; i++)
    {
        #pragma acc loop
        for(j = 0 ; j < col ; j++)
        {
            dst[i * col + j] = src1[i * col + j] * src2[i * col + j];
        }
    }
}

void sum_sqrt(double *src, double *dst, const int row, const int col)
{
    int i, j;
    
    #pragma acc parallel loop
    for(i = 0 ; i < row ; i++) 
    {
        #pragma acc loop
        for(j = 0 ; j < col ; j++) 
        {
            dst[j] += src[i * col + j];
        }
    }

    #pragma acc parallel loop
    for(i = 0; i < col ; i++) //i->col
    {
        dst[i] = sqrt(dst[i]);
    }
}

void update_activeMask(int *activeMask, double *residualNorms, double residualTolerance, int blocksize)
{
    int i;
    #pragma acc parallel loop
    for(i=0; i<blocksize; i++)
    {
        if((residualNorms[i] > residualTolerance) && activeMask[i] == 1)
            activeMask[i] = 1;
        else
            activeMask[i] = 0;
    }
}

void getActiveBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize)
{
    //activeBlockVectorR -> M*currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize

    int i, j, k=0;
    #pragma acc parallel loop firstprivate(k) private(j)
    for(i=0; i<M; i++)
    {
        k=0;
        for(j=0; j<blocksize; j++)
        {
             if(activeMask[j] == 1)
             {
                activeBlockVectorR[i*currentBlockSize+k] = blockVectorR[i*blocksize+j];
                k++;
             }
        }
    }
}
void updateBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize)
{
    //activeBlockVectorR -> M*currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize
    int i, j, k = 0;
    #pragma acc parallel loop firstprivate(k) private(j)
    for(i=0; i<M; i++)
    {
        k=0;
        for(j=0; j<blocksize; j++)
        {
             if(activeMask[j] == 1)
             {
                blockVectorR[i*blocksize+j]= activeBlockVectorR[i*currentBlockSize+k];
                k++;
             }
        }
    }
}
void mat_copy(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst)
{
    int i,j;
    #pragma acc parallel loop
    for(i=0; i<row; i++)
    {
        #pragma acc loop
        for(j=0; j<col; j++)
        {
            dst[(start_row+i)*ld_dst+(start_col+j)]=src[i*col+j];
        }

    }
}


void print_eigenvalues( int n, double* wr, double* wi ) 
{
    int j;
    for( j = 0; j < n; j++ ) 
    {
        if( wi[j] == (double)0.0 ) 
            printf( " %.15f", wr[j] );
        else 
            printf( " (%.15f,%6.2f)", wr[j], wi[j] );
    }
    printf( "\n" );
}

void custom_dlacpy(double *src, double *dst, int m, int n)
{
    //src[m*n] and dst[m*n]
    int i, j;
    #pragma acc parallel loop
    for(i = 0 ; i < m ; i++) //each row
    {
        #pragma acc loop
        for(j = 0 ; j < n ; j++) //each column
        {
            dst[i * n + j] = src[i * n + j];
        }
    }
}

template<typename T>
bool checkEquals( T* a, T* b, size_t outterSize, size_t innerSize)
{
    for(size_t i = 0 ; i < outterSize * innerSize ; ++i)
    {
        if(abs(a[i]-b[i])>0.2)
        {
            cout << i << " " << a[i] << " " << b[i] << endl;
            return false;
        }
    }
    return true;
}
template<typename T>
void csc2blkcoord(block<T> *&matrixBlock, T *xrem)
{
  int i, j, r, c, k, k1, k2, blkr, blkc, tmp;
  int **top;
  nrowblks = ceil(numrows/(float)(wblk));
  ncolblks = ceil(numcols/(float)(wblk));
  cout<<" nrowblks = "<<nrowblks<<endl;
  cout<<" ncolblks = "<<ncolblks<<endl;

  matrixBlock=new block<T>[nrowblks*ncolblks];
  top=new int*[nrowblks];

  for(i=0;i<nrowblks;i++)
  {
    top[i]=new int[ncolblks];
  }

  for(blkr=0;blkr<nrowblks;blkr++)
  {
    for(blkc=0;blkc<ncolblks;blkc++)
    {
      top[blkr][blkc]=0;
      matrixBlock[blkr*ncolblks+blkc].nnz=0;
    }
  }
  cout<<"Finish memory allocation for block.."<<endl;

  //cout<<"K1: "<<colptrs[0]<<" K2: "<<colptrs[1]<<endl;

  //cout<<"calculatig nnz per block"<<endl;

  //calculatig nnz per block
  for(c=0;c<numcols;c++)
  {
    k1=colptrs[c];
    k2=colptrs[c+1]-1;
    blkc=ceil((c+1)/(float)wblk);
    //cout<<"K1: "<<k1<<" K2: "<<k2<<" blkc: "<<blkc<<endl;

    for(k=k1-1;k<k2;k++)
    {
      r=irem[k];
      blkr=ceil(r/(float)wblk);
      matrixBlock[(blkr-1)*ncolblks+(blkc-1)].nnz++;
    }
  }

  //cout<<"allocating memory for each block"<<endl;

  for(blkc=0;blkc<ncolblks;blkc++)
  {
    for(blkr=0;blkr<nrowblks;blkr++)
    {
      //cout<<"br: "<<blkr<<" bc: "<<blkc<<" roffset: "<<blkr*wblk<<" coffset: "<<blkc*wblk<<endl;
      matrixBlock[blkr*ncolblks+blkc].roffset=blkr*wblk;
      matrixBlock[blkr*ncolblks+blkc].coffset=blkc*wblk;
      //cout<<"here 1"<<endl;

      if(matrixBlock[blkr*ncolblks+blkc].nnz>0)
      {
        matrixBlock[blkr*ncolblks+blkc].rloc=new unsigned short int[matrixBlock[blkr*ncolblks+blkc].nnz];
        matrixBlock[blkr*ncolblks+blkc].cloc=new unsigned short int[matrixBlock[blkr*ncolblks+blkc].nnz];
        //matrixBlock[blkr*ncolblks+blkc].val=new float[matrixBlock[blkr*ncolblks+blkc].nnz];
        matrixBlock[blkr*ncolblks+blkc].val=new T[matrixBlock[blkr*ncolblks+blkc].nnz];
      }
      else
      {
        matrixBlock[blkr*ncolblks+blkc].rloc=NULL;
        matrixBlock[blkr*ncolblks+blkc].cloc=NULL;
      }
    }
  }
  //cout<<"end for"<<endl;

  for(c=0;c<numcols;c++)
  {
    k1=colptrs[c];
    k2=colptrs[c+1]-1;
    blkc=ceil((c+1)/(float)wblk);

    for(k=k1-1;k<k2;k++)
    {
      r=irem[k];
      blkr=ceil(r/(float)wblk);

      matrixBlock[(blkr-1)*ncolblks+blkc-1].rloc[top[blkr-1][blkc-1]]=r - matrixBlock[(blkr-1)*ncolblks+blkc-1].roffset;
      matrixBlock[(blkr-1)*ncolblks+blkc-1].cloc[top[blkr-1][blkc-1]]=(c+1) -  matrixBlock[(blkr-1)*ncolblks+blkc-1].coffset;
      matrixBlock[(blkr-1)*ncolblks+blkc-1].val[top[blkr-1][blkc-1]]=xrem[k];

      top[blkr-1][blkc-1]=top[blkr-1][blkc-1]+1;
    }
  }

    //cout<<"end for 2"<<endl;

  //checking rloc and cloc values of each block
  /*for(blkr=0;blkr<nrowblks;blkr++)
  {
    for(blkc=0;blkc<ncolblks;blkc++)
    {
      if(matrixBlock[blkr*ncolblks+blkc].nnz>0)
      {
        cout<<"blkr: "<<blkr<<" blkc: "<<blkc<<" blk nnz: "<<matrixBlock[blkr*ncolblks+blkc].nnz<<endl;
        for(i=0;i<matrixBlock[blkr*ncolblks+blkc].nnz;i++)
        {
          cout<<matrixBlock[blkr*ncolblks+blkc].rloc[i]<<" ";
        }
        cout<<endl;
        for(i=0;i<matrixBlock[blkr*ncolblks+blkc].nnz;i++)
        {
          cout<<matrixBlock[blkr*ncolblks+blkc].cloc[i]<<" ";
        }
        cout<<endl;
        for(i=0;i<matrixBlock[blkr*ncolblks+blkc].nnz;i++)
        {
          cout<<matrixBlock[blkr*ncolblks+blkc].val[i]<<" ";
        }
        cout<<endl;
      }
    }
  }*/

  //delete top

  for(i=0;i<nrowblks;i++)
  {
    delete [] top[i];//=new int[ncolblks];
  }
  delete [] top;
}
template void csc2blkcoord<double>(block<double> *&matrixBlock, double *xrem); 

void spmm_csr(int row, int col, int nvec, int *row_ptr, int *col_index, double *value, double *Y, double *result)
{
    int i, j, k, start, end;
    int r, c, xcoef;

    //#pragma omp parallel for default(shared) private(start, end, r, c, xcoef, j, k)
    #pragma acc parallel loop private(start, end, r, c, xcoef, j, k)
    for(i = 0 ; i <= row ; i++)
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