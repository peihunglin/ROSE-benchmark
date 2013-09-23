/* 
   by Didem Unat 
   3D 7-point jacobi
   Written to be used as an input program to mint translator 
  
   See the alloc2D function, which allocates contiguous memory space to 
   the array. 
 */
//#include "common.h"
#include <stdio.h>
#include <math.h>
//#include <omp.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#define REAL double
#define FLOPS 8 
#define chunk 64
const double kMicro = 1.0e-6;

double ***alloc3D(int n,int m,int k)
{
  double ***m_buffer = ((void *)0);
  int nx = n;
  int ny = m;
  int nk = k;
  m_buffer = ((double ***)(malloc(sizeof(double **) * nk)));
  m_buffer?((void )0) : ((__assert_fail("m_buffer","heat3D.c",32,__PRETTY_FUNCTION__) , ((void )0)));
  double **m_tempzy = (double **)(malloc(sizeof(double *) * nk * ny));
  double *m_tempzyx = (double *)(malloc(sizeof(double ) * nx * ny * nk));
  int z;
  int y;
  for (z = 0; z < nk; (z++ , m_tempzy += ny)) {
    m_buffer[z] = m_tempzy;
    for (y = 0; y < ny; (y++ , m_tempzyx += nx)) {
      m_buffer[z][y] = m_tempzyx;
    }
  }
  return m_buffer;
}

double getTime()
{
  struct timeval TV;
  const int RC = gettimeofday(&TV,((void *)0));
  if (RC == -1) {
    printf("ERROR: Bad call to gettimeofday\n");
    return (-1);
  }
  return ((double )TV . tv_sec) + kMicro * ((double )TV . tv_usec);
// end getTime()                                                                               
}
//allocate 3D array                                                                                     

double ***alloc3D_(int n,int m,int k)
{
  double ***E = ((void *)0);
  int nx = n;
  int ny = m;
  int nk = k;
  E = ((double ***)(malloc(sizeof(double **) * nk)));
  E?((void )0) : ((__assert_fail("E","heat3D.c",71,__PRETTY_FUNCTION__) , ((void )0)));
  E[0] = ((double **)(malloc(sizeof(double *) * nk * ny)));
  E[0][0] = ((double *)(malloc(sizeof(double ) * nx * ny * nk)));
  int jj;
  int kk;
  for (kk = 0; kk < nk; kk++) {
    if (kk > 0) {
      E[kk] = E[kk - 1] + ny;
      E[kk][0] = E[kk - 1][0] + ny * nx;
    }
    for (jj = 1; jj < ny; jj++) {
      E[kk][jj] = E[kk][jj - 1] + nx;
    }
  }
  return E;
}

void free3D(double ***E)
{
//int k=0;
/*  for(k=0 ; k < m ; k++)
    {
      free(E[k]);
      }*/
  free(E[0][0]);
  free(E[0]);
  free(E);
}

void init(double ***E,int N,int M,int K)
{
  int i;
  int j;
  int k;
  for (k = 0; k < K; k++) 
    for (i = 0; i < M; i++) 
      for (j = 0; j < N; j++) {
        E[k][i][j] = 1.0;
        if (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1) {
          E[k][i][j] = 0.0;
        }
      }
}
//calculate l2norm for comparison

void calculatel2Norm(double ***E,int N,int M,int K,int nIters)
{
  int i;
  int j;
  int k = 0;
  float mx = (-1);
  float l2norm = 0;
  for (k = 1; k <= K; k++) {
    for (j = 1; j <= M; j++) {
      for (i = 1; i <= N; i++) {
        l2norm += E[k][j][i] * E[k][j][i];
        if (E[k][j][i] > mx) {
          mx = E[k][j][i];
        }
      }
    }
  }
  l2norm /= ((float )(N * M * K));
  l2norm = (sqrt(l2norm));
  printf(":N %d M %d K %d , iteration %d\n",N,M,K,nIters);
  printf(":max: %20.12e, l2norm: %20.12e\n",mx,l2norm);
}
static void mint_1_1527(int n,int m,int k,double c0,double c1,cudaPitchedPtr dev_2_Unew,cudaPitchedPtr dev_1_Uold,int num2blockDim_1_1527,float invYnumblockDim_1_1527);

int main(int argc,char *argv[])
{
  int n = 256;
  int m = 256;
  int k = 256;
  double c0 = 0.5;
  double c1 = -0.25;
  double ***Unew;
  double ***Uold;
  Unew = alloc3D(n + 2,m + 2,k + 2);
  Uold = alloc3D(n + 2,m + 2,k + 2);
  init(Unew,n + 2,m + 2,k + 2);
  init(Uold,n + 2,m + 2,k + 2);
  int T = 20;
  printf("\n=====Timings (sec) for 7-Point Jacobi, Solving Heat Eqn ");
  if (sizeof(double ) == 4) {
    printf(" (Single Precision) =====\n");
  }
  if (sizeof(double ) == 8) {
    printf(" (Double Precision) =====\n");
  }
  printf("Kernel\t Time(sec)\tGflops  \tBW-ideal(GB/s)\tBW-algorithm (N=(%d,%d) iters=%d)\n",n,n,T);
  printf("------\t----------\t--------\t--------------\t------------\n");
  int nIters = 0;
  double time_elapsed;
  double Gflops = 0.0;
/* Mint: Replaced Pragma: #pragma mint copy( Uold, toDevice,( n+2 ),( m+2 ),( k+2 ) ) */
  cudaError_t stat_dev_1_Uold;
  cudaExtent ext_dev_1_Uold = make_cudaExtent(((n+2)) * sizeof(double ),((m+2)),((k+2))));
/* Mint: Malloc on the device */
  cudaPitchedPtr dev_1_Uold;
  stat_dev_1_Uold = cudaMalloc3D(&dev_1_Uold,ext_dev_1_Uold);
  if (stat_dev_1_Uold != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_1_Uold));
/* Mint: Copy host to device */
  cudaMemcpy3DParms param_1_dev_1_Uold = {0};
  param_1_dev_1_Uold . srcPtr = make_cudaPitchedPtr(((void *)Uold[0][0]),((n+2)) * sizeof(double ),((n+2)),((m+2)));
  param_1_dev_1_Uold . dstPtr = dev_1_Uold;
  param_1_dev_1_Uold . extent = ext_dev_1_Uold;
  param_1_dev_1_Uold . kind = cudaMemcpyHostToDevice;
  stat_dev_1_Uold = cudaMemcpy3D(&param_1_dev_1_Uold);
  if (stat_dev_1_Uold != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_1_Uold));
/* Mint: Replaced Pragma: #pragma mint copy( Unew, toDevice,( n+2 ), m+2,( k+2 ) ) */
  cudaError_t stat_dev_2_Unew;
  cudaExtent ext_dev_2_Unew = make_cudaExtent(((n+2)) * sizeof(double ),(m+2),((k+2))));
/* Mint: Malloc on the device */
  cudaPitchedPtr dev_2_Unew;
  stat_dev_2_Unew = cudaMalloc3D(&dev_2_Unew,ext_dev_2_Unew);
  if (stat_dev_2_Unew != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_2_Unew));
/* Mint: Copy host to device */
  cudaMemcpy3DParms param_2_dev_2_Unew = {0};
  param_2_dev_2_Unew . srcPtr = make_cudaPitchedPtr(((void *)Unew[0][0]),((n+2)) * sizeof(double ),((n+2)),(m+2));
  param_2_dev_2_Unew . dstPtr = dev_2_Unew;
  param_2_dev_2_Unew . extent = ext_dev_2_Unew;
  param_2_dev_2_Unew . kind = cudaMemcpyHostToDevice;
  stat_dev_2_Unew = cudaMemcpy3D(&param_2_dev_2_Unew);
  if (stat_dev_2_Unew != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_2_Unew));
{
    time_elapsed = getTime();
    int t = 0;
    while(t < T){
      t++;
      int x;
      int y;
      int z;
//7-point stencil
      
#pragma mint for nest(all) tile(16,16,16) chunksize(1,1,16)
      int num3blockDim_1_1527 = (k - 1 + 1) % 16 == 0?(k - 1 + 1) / 16 : (k - 1 + 1) / 16 + 1;
      int num2blockDim_1_1527 = (m - 1 + 1) % 16 == 0?(m - 1 + 1) / 16 : (m - 1 + 1) / 16 + 1;
      int num1blockDim_1_1527 = (n - 1 + 1) % 16 == 0?(n - 1 + 1) / 16 : (n - 1 + 1) / 16 + 1;
      float invYnumblockDim_1_1527 = 1.00000F / num2blockDim_1_1527;
      dim3 blockDim_1_1527(16,16,1);
      dim3 gridDim_1_1527(num1blockDim_1_1527,num2blockDim_1_1527*num3blockDim_1_1527);
      mint_1_1527<<<gridDim_1_1527,blockDim_1_1527>>>(n,m,k,c0,c1,dev_2_Unew,dev_1_Uold,num2blockDim_1_1527,invYnumblockDim_1_1527);
      cudaThreadSynchronize();
      cudaError_t err_mint_1_1527 = cudaGetLastError();
      if (err_mint_1_1527) {
        fprintf(stderr,"In %s, %s\n","mint_1_1527",cudaGetErrorString(err_mint_1_1527));
      }
      
#pragma mint single
{
        double ***tmp;
        void *dev_tmp;
        dev_tmp = dev_1_Uold . ptr;
        dev_1_Uold . ptr = dev_2_Unew . ptr;
        dev_2_Unew . ptr = dev_tmp;
        nIters = t;
      }
//end of while
    }
//end of parallel region
  }
  cudaFree(dev_2_Unew . ptr);
  cudaFree(dev_1_Uold . ptr);
  
#pragma mint copy(Uold, fromDevice, (n+2), (m+2), (k+2))
  time_elapsed = getTime() - time_elapsed;
  Gflops = ((double )((nIters * n * m * k) * 1.0e-9 * 8)) / time_elapsed;
  printf("%s%3.3f \t%5.3f\n","Heat3D   ",time_elapsed,Gflops);
  calculatel2Norm(Uold,n,m,k,T);
  free3D(Uold);
  free3D(Unew);
  return 0;
}

__global__ static void mint_1_1527(int n,int m,int k,double c0,double c1,cudaPitchedPtr dev_2_Unew,cudaPitchedPtr dev_1_Uold,int num2blockDim_1_1527,float invYnumblockDim_1_1527)
{
#define TILE_X 16
#define TILE_Y 16
  __device__ __shared__ double _sh_block_Uold[TILE_Y + 2][TILE_X + 2];
  double *Unew = (double *)dev_2_Unew . ptr;
  int _width = dev_2_Unew . pitch / sizeof(double );
  int _slice = dev_2_Unew . ysize * _width;
  double *Uold = (double *)dev_1_Uold . ptr;
  float blocksInY = num2blockDim_1_1527;
  float invBlocksInY = invYnumblockDim_1_1527;
  int _p_x;
  int _p_y;
  int _p_z;
{
    double up__rUold = Uold[_index3D - _slice];
    double _rUold = Uold[_index3D];
    int _upperb_y = m;
    int _upperb_x = n;
    int _idx = threadIdx.x + 1;
    int _gidx = _idx + blockDim.x * blockIdx.x;
    int _idy = threadIdx.y + 1;
    int _gidy = _idy + blockDim.y * 1 * blockIdx.y;
    int _idz = threadIdx.z + 1;
    int blockIdxz = blockIdx.y * invBlocksInY;
    int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
    _gidy = _idy + blockIdxy * blockDim.y;
    int _gidz = _idz + blockIdxz * 16;
    int _index3D = _gidx + _gidy * _width + _gidz * _slice;
    _idz = 1;
    _idy = threadIdx.y + 1;
    _idx = threadIdx.x + 1;
    int _borderIdx = _idx;
    int _borderIdy = 0;
    int _borderGlobalIndexDiff = 0;
    _borderIdx = (threadIdx.y == 1?0 : _borderIdx);
    _borderIdx = (threadIdx.y == 2?blockDim.x + 1 : _borderIdx);
    _borderIdy = (threadIdx.y == 3?blockDim.y + 1 : _borderIdy);
    _borderIdy = (threadIdx.y == 1 || threadIdx.y == 2?_idx : _borderIdy);
    _borderGlobalIndexDiff = _borderIdx - _idx + _width * (_borderIdy - _idy);
{
      int _upper_gidz = _gidz + 16 < k?_gidz + 15 : k;
{
        if (_gidy >= 1 && _gidy <= m) {{
            if (_gidx >= 1 && _gidx <= n) 
              for (_gidz = _gidz; _gidz <= _upper_gidz; _gidz += 1) {
                _index3D = _gidx + _gidy * _width + _gidz * _slice;
{
                  _sh_block_Uold[_idy][_idx] = _rUold;
                  if (threadIdx.y < 4 * 1) 
                    _sh_block_Uold[_borderIdy][_borderIdx] = Uold[_index3D + _borderGlobalIndexDiff];
                  double down__rUold = Uold[_index3D + _slice];
                  double _rUnew;
                  __syncthreads();
                  _rUnew = c0 * _sh_block_Uold[_idy][_idx] + c1 * (_sh_block_Uold[_idy][_idx - 1] + _sh_block_Uold[_idy][_idx + 1] + _sh_block_Uold[_idy - 1][_idx] + _sh_block_Uold[_idy + 1][_idx] + up__rUold + down__rUold);
                  Unew[_index3D] = _rUnew;
                  up__rUold = _rUold;
                  _rUold = down__rUold;
                  __syncthreads();
                }
              }
          }
        }
      }
    }
  }
}
