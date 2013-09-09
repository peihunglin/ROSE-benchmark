#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// Add timing support
#include <sys/time.h>

double time_stamp()
{
  struct timeval t;
  double time;
  gettimeofday(&t,((struct timezone *)((void *)0)));
  time = t . tv_sec + 1.0e-6 * t . tv_usec;
  return time;
}
double time1;
double time2;
void driver();
void initialize();
void jacobi();
void error_check();
/************************************************************
* program to solve a finite difference 
* discretization of Helmholtz equation :  
* (d2/dx2)u + (d2/dy2)u - alpha u = f 
* using Jacobi iterative method. 
*
* Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
* Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
*
* This c version program is translated by 
* Chunhua Liao, University of Houston, Jan, 2005 
* 
* Directives are used in this code to achieve parallelism. 
* All do loops are parallelized with default 'static' scheduling.
* 
* Input :  n - grid dimension in x direction 
*          m - grid dimension in y direction
*          alpha - Helmholtz constant (always greater than 0.0)
*          tol   - error tolerance for iterative solver
*          relax - Successice over relaxation parameter
*          mits  - Maximum iterations for iterative solver
*
* On output 
*       : u(n,m) - Dependent variable (solutions)
*       : f(n,m) - Right hand side function 
*************************************************************/
#define MSIZE 18 
int n;
int m;
int mits;
#define REAL float // flexible between float and double
float tol;
float relax = 1.0;
float alpha = 0.0543;
float u[18][18];
float f[18][18];
float uold[18][18];
float dx;
float dy;

int main()
{
//  float toler;
/*      printf("Input n,m (< %d) - grid dimension in x,y direction:\n",MSIZE); 
          scanf ("%d",&n);
          scanf ("%d",&m);
          printf("Input tol - error tolerance for iterative solver\n"); 
          scanf("%f",&toler);
          tol=(double)toler;
          printf("Input mits - Maximum iterations for solver\n"); 
          scanf("%d",&mits);
          */
  n = 18;
  m = 18;
  tol = 0.0000000001;
  mits = 5000;
#if 0 // Not yet support concurrent CPU and GPU threads  
#ifdef _OPENMP
#endif
#endif  
  driver();
  return 0;
}
/*************************************************************
* Subroutine driver () 
* This is where the arrays are allocated and initialzed. 
*
* Working varaibles/arrays 
*     dx  - grid spacing in x direction 
*     dy  - grid spacing in y direction 
*************************************************************/

void driver()
{
  initialize();
  time1 = time_stamp();
/* Solve Helmholtz equation */
  jacobi();
  time2 = time_stamp();
  printf("------------------------\n");
  printf("Execution time = %f\n",time2 - time1);
/* error_check (n,m,alpha,dx,dy,u,f)*/
  error_check();
}
/*      subroutine initialize (n,m,alpha,dx,dy,u,f) 
******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/

void initialize()
{
  int i;
  int j;
  int xx;
  int yy;
{
    int c1;
    int c2;
    if (m >= 1) {
      for (c1 = 0; c1 <= n + -1; c1++) {
        for (c2 = 0; c2 <= m + -1; c2++) {
          u[c1][c2] = 0.0;
        }
      }
    }
    dx = (2.0 / (n - 1));
    dy = (2.0 / (m - 1));
    if (m >= 1) {
      for (c1 = 0; c1 <= n + -1; c1++) {
        for (c2 = 0; c2 <= m + -1; c2++) {
          xx = ((int )(-1.0 + (dx * (c1 - 1))));
          yy = ((int )(-1.0 + (dy * (c2 - 1))));
          f[c1][c2] = (-1.0 * alpha * (1.0 - (xx * xx)) * (1.0 - (yy * yy)) - 2.0 * (1.0 - (xx * xx)) - 2.0 * (1.0 - (yy * yy)));
        }
      }
    }
  }
}
/*      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,maxit)
******************************************************************
* Subroutine HelmholtzJ
* Solves poisson equation on rectangular grid assuming : 
* (1) Uniform discretization in each direction, and 
* (2) Dirichlect boundary conditions 
* 
* Jacobi method is used in this routine 
*
* Input : n,m   Number of grid points in the X/Y directions 
*         dx,dy Grid spacing in the X/Y directions 
*         alpha Helmholtz eqn. coefficient 
*         omega Relaxation factor 
*         f(n,m) Right hand side function 
*         u(n,m) Dependent variable/Solution
*         tol    Tolerance for iterative solver 
*         maxit  Maximum number of iterations 
*
* Output : u(n,m) - Solution 
*****************************************************************/

void jacobi()
{
  float omega;
  int i;
  int j;
  int k;
  float error;
  float resid;
  float ax;
  float ay;
  float b;
{
//      double  error_local;
//      float ta,tb,tc,td,te,ta1,ta2,tb1,tb2,tc1,tc2,td1,td2;
//      float te1,te2;
//      float second;
    omega = relax;
/*
   * Initialize coefficients */
/* X-direction coef */
    ax = (1.0 / (dx * dx));
/* Y-direction coef */
    ay = (1.0 / (dy * dy));
/* Central coeff */
    b = (-2.0 / (dx * dx) - 2.0 / (dy * dy) - alpha);
    error = (10.0 * tol);
    k = 1;
  }
// An optimization on top of naive coding: promoting data handling outside the while loop
// data properties may change since the scope is bigger:
  
#pragma omp target data map(in:n, m, omega, ax, ay, b, f[0:n][0:m]) map(inout:u[0:n][0:m]) map(alloc:uold[0:n][0:m])
  while(k <= mits && error > tol){{
      error = 0.0;
    }
/* Copy new solution into old */
//#pragma omp parallel
//    {
//map(in:n, m, u[0:n][0:m]) map(out:uold[0:n][0:m])
    
#pragma omp target
    
#pragma omp parallel for private(j,i)
{
      int c1;
      int c0;
      if (m >= 1 && n >= 1) {
        for (c0 = 0; c0 <= n + -1; c0++) {
          for (c1 = 0; c1 <= m + -1; c1++) {
            uold[c0][c1] = u[c0][c1];
          }
        }
      }
    }
//map(in:n, m, omega, ax, ay, b, f[0:n][0:m], uold[0:n][0:m]) map(out:u[0:n][0:m])
    
#pragma omp target
// nowait  collapse(2) 
    
#pragma omp parallel for private(resid,j,i) reduction(+:error)
{
      int c1;
      int c0;
      if (m >= 3 && n >= 3) {
        for (c0 = 1; c0 <= n + -2; c0++) {
          for (c1 = 1; c1 <= m + -2; c1++) {
            resid = (ax * (uold[c0 - 1][c1] + uold[c0 + 1][c1]) + ay * (uold[c0][c1 - 1] + uold[c0][c1 + 1]) + b * uold[c0][c1] - f[c0][c1]) / b;
            error = error + resid * resid;
            u[c0][c1] = uold[c0][c1] - omega * resid;
          }
        }
      }
    }
//    }
/*  omp end parallel */
/* Error check */
    if (k % 500 == 0) {
      printf("Finished %d iteration with error =%f\n",k,error);
    }
    error = (sqrt(error) / (n * m));
{
      k = k + 1;
    }
/*  End iteration loop */
  }
  printf("Total Number of Iterations:%d\n",k);
  printf("Residual:%E\n",error);
}
/*      subroutine error_check (n,m,alpha,dx,dy,u,f) 
      implicit none 
************************************************************
* Checks error between numerical and exact solution 
*
************************************************************/

void error_check()
{
  int i;
  int j;
  float xx;
  float yy;
  float temp;
  float error;
{
    int c1;
    int c2;
    dx = (2.0 / (n - 1));
    dy = (2.0 / (m - 1));
    error = 0.0;
    if (m >= 1) {
      for (c1 = 0; c1 <= n + -1; c1++) {
        for (c2 = 0; c2 <= m + -1; c2++) {
          xx = (-1.0 + (dx * (c1 - 1)));
          yy = (-1.0 + (dy * (c2 - 1)));
          temp = (u[c1][c2] - (1.0 - (xx * xx)) * (1.0 - (yy * yy)));
          error = error + temp * temp;
        }
      }
    }
  }
  error = (sqrt(error) / (n * m));
  printf("Solution Error :%E \n",error);
}
