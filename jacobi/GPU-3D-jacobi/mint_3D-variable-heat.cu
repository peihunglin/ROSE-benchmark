/* 
   Written by Xing Cai, Nov-2010
   Modified by Didem Unat, Dec-2010
   A test program for solving the 3D heat conduction problem
         -div ( kappa(x,y,z) grad u ) = f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z).
   Note that the heat conduction coefficient is a variable field.
   An explicit time stepping scheme is used.
*/
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
//#include <omp.h>
#include <sys/time.h>
#include <assert.h>
#define FLOPS 26.0
#define REAL double
#define chunk 64 
const double kMicro = 1.0e-6;

double ***alloc3D(int n,int m,int k)
{
  double ***m_buffer = ((void *)0);
  int nx = n;
  int ny = m;
  int nk = k;
  m_buffer = ((double ***)(malloc(sizeof(double **) * nk)));
  m_buffer?((void )0) : ((__assert_fail("m_buffer","3D-variable-heat.c",36,__PRETTY_FUNCTION__) , ((void )0)));
  double **m_tempzy = (double **)(malloc(sizeof(double *) * nk * ny));
  double *m_tempzyx = (double *)(malloc(sizeof(double ) * nx * ny * nk));
  for (int z = 0; z < nk; (z++ , m_tempzy += ny)) {
    m_buffer[z] = m_tempzy;
    for (int y = 0; y < ny; (y++ , m_tempzyx += nx)) {
      m_buffer[z][y] = m_tempzyx;
    }
  }
  return m_buffer;
}

void free3D(double ***E)
{
  free(E[0][0]);
  free(E[0]);
  free(E);
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
static void mint_1_1527(int n,cudaPitchedPtr dev_1_u_old,cudaPitchedPtr dev_2_u_new,cudaPitchedPtr dev_3_rhs,cudaPitchedPtr dev_4_kappa,double factor,int num2blockDim_1_1527,float invYnumblockDim_1_1527);

int main(int nargs,char **args)
{
/* number of points in each direction */
  int n;
/* grid spacing, same in all the directions */
  double h;
  double ***u_old;
  double ***u_new;
  double ***rhs;
  double ***kappa;
  double factor;
  double l2_norm;
  double dt;
  double T;
  int i;
  int j;
  int k;
  int max_iters = 20;
  if (nargs > 1) {
    n = atoi(args[1]);
  }
  else {
    n = 256;
  }
  T = 1.0;
  h = 1.0 / (n - 1);
/* a safe choice of dt when k(x,y,z)<1 */
  dt = h * h / 6.0;
  u_old = alloc3D(n + 2,n + 2,n + 2);
  u_new = alloc3D(n + 2,n + 2,n + 2);
  rhs = alloc3D(n + 2,n + 2,n + 2);
  kappa = alloc3D(n + 2,n + 2,n + 2);
/* fill values of rhs */
  for (k = 0; k <= n + 1; k++) 
    for (j = 0; j <= n + 1; j++) 
      for (i = 0; i <= n + 1; i++) 
        rhs[k][j][i] = dt * (1.0 + cos(3.14159265358979323846 * i * h) * cos(3.14159265358979323846 * j * h) * cos(3.14159265358979323846 * k * h));
/* fill values of kappa */
  for (k = 0; k <= n + 1; k++) 
    for (j = 0; j <= n + 1; j++) 
      for (i = 0; i <= n + 1; i++) 
        kappa[k][j][i] = 0.5 + 0.45 * sin(3.14159265358979323846 * i * h) * sin(3.14159265358979323846 * j * h) * sin(3.14159265358979323846 * k * h);
/* fill initial values */
  for (k = 0; k <= n + 1; k++) 
    for (j = 0; j <= n + 1; j++) 
      for (i = 0; i <= n + 1; i++) {
        u_old[k][j][i] = sin(3.14159265358979323846 * i * h) * sin(3.14159265358979323846 * j * h) * sin(3.14159265358979323846 * k * h);
        u_new[k][j][i] = 0.;
      }
/* main time loop */
  int nIters = 0;
  factor = dt / h / h / 2.0;
  printf("\n=====Timings (sec) for 3D variable-coefficient Eqn-7 point ");
  if (sizeof(double ) == 4) {
    printf(" (Single Precision) =====\n");
  }
  if (sizeof(double ) == 8) {
    printf(" (Double Precision) =====\n");
  }
  printf("Kernel\t Time(sec)\tGflops  \tBW-ideal(GB/s)\tBW-algorithm (N=(%d) iters=%d)\n",n,max_iters);
  printf("------\t----------\t--------\t--------------\t------------\n");
  double time_elapsed = getTime();
  double Gflops = 0.0;
/* Mint: Replaced Pragma: #pragma mint copy( u_old, toDevice,( n+2 ), n+2,( n+2 )) */
  cudaError_t stat_dev_1_u_old;
  cudaExtent ext_dev_1_u_old = make_cudaExtent(((n+2)) * sizeof(double ),(n+2),((n+2))));
/* Mint: Malloc on the device */
  cudaPitchedPtr dev_1_u_old;
  stat_dev_1_u_old = cudaMalloc3D(&dev_1_u_old,ext_dev_1_u_old);
  if (stat_dev_1_u_old != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_1_u_old));
/* Mint: Copy host to device */
  cudaMemcpy3DParms param_1_dev_1_u_old = {0};
  param_1_dev_1_u_old . srcPtr = make_cudaPitchedPtr(((void *)u_old[0][0]),((n+2)) * sizeof(double ),((n+2)),(n+2));
  param_1_dev_1_u_old . dstPtr = dev_1_u_old;
  param_1_dev_1_u_old . extent = ext_dev_1_u_old;
  param_1_dev_1_u_old . kind = cudaMemcpyHostToDevice;
  stat_dev_1_u_old = cudaMemcpy3D(&param_1_dev_1_u_old);
  if (stat_dev_1_u_old != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_1_u_old));
/* Mint: Replaced Pragma: #pragma mint copy( u_new, toDevice,( n+2 ), n+2,( n+2 )) */
  cudaError_t stat_dev_2_u_new;
  cudaExtent ext_dev_2_u_new = make_cudaExtent(((n+2)) * sizeof(double ),(n+2),((n+2))));
/* Mint: Malloc on the device */
  cudaPitchedPtr dev_2_u_new;
  stat_dev_2_u_new = cudaMalloc3D(&dev_2_u_new,ext_dev_2_u_new);
  if (stat_dev_2_u_new != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_2_u_new));
/* Mint: Copy host to device */
  cudaMemcpy3DParms param_2_dev_2_u_new = {0};
  param_2_dev_2_u_new . srcPtr = make_cudaPitchedPtr(((void *)u_new[0][0]),((n+2)) * sizeof(double ),((n+2)),(n+2));
  param_2_dev_2_u_new . dstPtr = dev_2_u_new;
  param_2_dev_2_u_new . extent = ext_dev_2_u_new;
  param_2_dev_2_u_new . kind = cudaMemcpyHostToDevice;
  stat_dev_2_u_new = cudaMemcpy3D(&param_2_dev_2_u_new);
  if (stat_dev_2_u_new != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_2_u_new));
/* Mint: Replaced Pragma: #pragma mint copy( rhs, toDevice,( n+2 ), n+2,( n+2 )) */
  cudaError_t stat_dev_3_rhs;
  cudaExtent ext_dev_3_rhs = make_cudaExtent(((n+2)) * sizeof(double ),(n+2),((n+2))));
/* Mint: Malloc on the device */
  cudaPitchedPtr dev_3_rhs;
  stat_dev_3_rhs = cudaMalloc3D(&dev_3_rhs,ext_dev_3_rhs);
  if (stat_dev_3_rhs != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_3_rhs));
/* Mint: Copy host to device */
  cudaMemcpy3DParms param_3_dev_3_rhs = {0};
  param_3_dev_3_rhs . srcPtr = make_cudaPitchedPtr(((void *)rhs[0][0]),((n+2)) * sizeof(double ),((n+2)),(n+2));
  param_3_dev_3_rhs . dstPtr = dev_3_rhs;
  param_3_dev_3_rhs . extent = ext_dev_3_rhs;
  param_3_dev_3_rhs . kind = cudaMemcpyHostToDevice;
  stat_dev_3_rhs = cudaMemcpy3D(&param_3_dev_3_rhs);
  if (stat_dev_3_rhs != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_3_rhs));
/* Mint: Replaced Pragma: #pragma mint copy( kappa, toDevice,( n+2 ), n+2,( n+2 )) */
  cudaError_t stat_dev_4_kappa;
  cudaExtent ext_dev_4_kappa = make_cudaExtent(((n+2)) * sizeof(double ),(n+2),((n+2))));
/* Mint: Malloc on the device */
  cudaPitchedPtr dev_4_kappa;
  stat_dev_4_kappa = cudaMalloc3D(&dev_4_kappa,ext_dev_4_kappa);
  if (stat_dev_4_kappa != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_4_kappa));
/* Mint: Copy host to device */
  cudaMemcpy3DParms param_4_dev_4_kappa = {0};
  param_4_dev_4_kappa . srcPtr = make_cudaPitchedPtr(((void *)kappa[0][0]),((n+2)) * sizeof(double ),((n+2)),(n+2));
  param_4_dev_4_kappa . dstPtr = dev_4_kappa;
  param_4_dev_4_kappa . extent = ext_dev_4_kappa;
  param_4_dev_4_kappa . kind = cudaMemcpyHostToDevice;
  stat_dev_4_kappa = cudaMemcpy3D(&param_4_dev_4_kappa);
  if (stat_dev_4_kappa != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_4_kappa));
{
    int iters = 0;
    double t = 0.0;
    while(t < T && iters < max_iters){
      t += dt;
      ++iters;
/* update each interior point */
      
#pragma mint for nest(all) tile ( 16, 16, 1 )
      int num3blockDim_1_1527 = (n - 1 + 1) % 1 == 0?(n - 1 + 1) / 1 : (n - 1 + 1) / 1 + 1;
      int num2blockDim_1_1527 = (n - 1 + 1) % 16 == 0?(n - 1 + 1) / 16 : (n - 1 + 1) / 16 + 1;
      int num1blockDim_1_1527 = (n - 1 + 1) % 16 == 0?(n - 1 + 1) / 16 : (n - 1 + 1) / 16 + 1;
      float invYnumblockDim_1_1527 = 1.00000F / num2blockDim_1_1527;
      dim3 blockDim_1_1527(16,16,1);
      dim3 gridDim_1_1527(num1blockDim_1_1527,num2blockDim_1_1527*num3blockDim_1_1527);
      mint_1_1527<<<gridDim_1_1527,blockDim_1_1527>>>(n,dev_1_u_old,dev_2_u_new,dev_3_rhs,dev_4_kappa,factor,num2blockDim_1_1527,invYnumblockDim_1_1527);
      cudaThreadSynchronize();
      cudaError_t err_mint_1_1527 = cudaGetLastError();
      if (err_mint_1_1527) {
        fprintf(stderr,"In %s, %s\n","mint_1_1527",cudaGetErrorString(err_mint_1_1527));
      }
/* pointer swap */
      
#pragma mint single
{
        double ***tmp;
        void *dev_tmp;
        dev_tmp = dev_1_u_old . ptr;
        dev_1_u_old . ptr = dev_2_u_new . ptr;
        dev_2_u_new . ptr = dev_tmp;
        nIters = iters;
      }
    }
  }
/* Mint: Replaced Pragma: #pragma mint copy( u_old, fromDevice,( n+2 ),( n+2 ),( n+2 )) */
/* Mint: Copy device to host */
  cudaMemcpy3DParms param_5_dev_1_u_old = {0};
  param_5_dev_1_u_old . srcPtr = dev_1_u_old;
  param_5_dev_1_u_old . dstPtr = make_cudaPitchedPtr(((void *)u_old[0][0]),((n+2)) * sizeof(double ),((n+2)),((n+2)));
  param_5_dev_1_u_old . extent = ext_dev_1_u_old;
  param_5_dev_1_u_old . kind = cudaMemcpyDeviceToHost;
  stat_dev_1_u_old = cudaMemcpy3D(&param_5_dev_1_u_old);
  if (stat_dev_1_u_old != cudaSuccess) 
    fprintf(stderr,"%s\n",cudaGetErrorString(stat_dev_1_u_old));
  cudaFree(dev_1_u_old . ptr);
  cudaFree(dev_2_u_new . ptr);
  cudaFree(dev_3_rhs . ptr);
  cudaFree(dev_4_kappa . ptr);
  time_elapsed = getTime() - time_elapsed;
  Gflops = ((double )((nIters * n * n * n) * 1.0e-9 * 26.0)) / time_elapsed;
  l2_norm = 0;
  for (k = 0; k <= n + 1; k++) 
    for (j = 0; j <= n + 1; j++) 
      for (i = 0; i <= n + 1; i++) {
        factor = sin(3.14159265358979323846 * i * h) * sin(3.14159265358979323846 * j * h) * sin(3.14159265358979323846 * k * h);
        l2_norm += (factor - u_old[k][j][i]) * (factor - u_old[k][j][i]);
      }
  printf("%s%3.3f \t%5.3f\n","VariableHeat   ",time_elapsed,Gflops);
  printf(":N %d M %d K %d , iteration %d\n",n,n,n,nIters);
  printf(":max: %20.12e, l2norm: %20.12e\n",factor,sqrt(l2_norm * h * h * h));
  free3D(u_new);
  free3D(u_old);
  free3D(rhs);
  free3D(kappa);
  return 0;
}

__global__ static void mint_1_1527(int n,cudaPitchedPtr dev_1_u_old,cudaPitchedPtr dev_2_u_new,cudaPitchedPtr dev_3_rhs,cudaPitchedPtr dev_4_kappa,double factor,int num2blockDim_1_1527,float invYnumblockDim_1_1527)
{
#define TILE_X 16
#define TILE_Y 16
  __device__ __shared__ double _sh_block_kappa[TILE_Y + 2][TILE_X + 2];
  __device__ __shared__ double _sh_block_u_old[TILE_Y + 2][TILE_X + 2];
  double *u_old = (double *)dev_1_u_old . ptr;
  int _width = dev_1_u_old . pitch / sizeof(double );
  int _slice = dev_1_u_old . ysize * _width;
  double *u_new = (double *)dev_2_u_new . ptr;
  double *rhs = (double *)dev_3_rhs . ptr;
  double *kappa = (double *)dev_4_kappa . ptr;
  float blocksInY = num2blockDim_1_1527;
  float invBlocksInY = invYnumblockDim_1_1527;
  int _p_i;
  int _p_j;
  int _p_k;
{
    int _upperb_y = n;
    int _upperb_x = n;
    int _idx = threadIdx.x + 1;
    int _gidx = _idx + blockDim.x * blockIdx.x;
    int _idy = threadIdx.y + 1;
    int _gidy = _idy + blockDim.y * 1 * blockIdx.y;
    int _idz = threadIdx.z + 1;
    int blockIdxz = blockIdx.y * invBlocksInY;
    int blockIdxy = blockIdx.y - blockIdxz * blocksInY;
    _gidy = _idy + blockIdxy * blockDim.y;
    int _gidz = _idz + blockIdxz * blockDim.z;
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
      if (_gidz >= 1 && _gidz <= n) {{{
            if (_gidy >= 1 && _gidy <= n) {{{
                  if (_gidx >= 1 && _gidx <= n) {{
                      if (threadIdx.y < 4 * 1) 
                        _sh_block_kappa[_borderIdy][_borderIdx] = kappa[_index3D + _borderGlobalIndexDiff];
                      if (threadIdx.y < 4 * 1) 
                        _sh_block_u_old[_borderIdy][_borderIdx] = u_old[_index3D + _borderGlobalIndexDiff];
                      double _rrhs = rhs[_index3D];
                      double _ru_new;
                      double _rkappa = kappa[_index3D];
                      _sh_block_kappa[_idy][_idx] = _rkappa;
                      double _ru_old = u_old[_index3D];
                      _sh_block_u_old[_idy][_idx] = _ru_old;
                      __syncthreads();
                      _ru_new = _ru_old + _rrhs + factor * ((_sh_block_kappa[_idy][_idx + 1] + _rkappa) * (_sh_block_u_old[_idy][_idx + 1] - _ru_old) + (_rkappa + _sh_block_kappa[_idy][_idx - 1]) * (_ru_old - _sh_block_u_old[_idy][_idx - 1]) + (_sh_block_kappa[_idy + 1][_idx] + _rkappa) * (_sh_block_u_old[_idy + 1][_idx] - _ru_old) + (_rkappa + _sh_block_kappa[_idy - 1][_idx]) * (_ru_old - _sh_block_u_old[_idy - 1][_idx]) + (kappa[_index3D + _slice] + _rkappa) * (u_old[_index3D + _slice] - _ru_old) + (_rkappa + kappa[_index3D - _slice]) * (_ru_old - u_old[_index3D - _slice]));
                      u_new[_index3D] = _ru_new;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
