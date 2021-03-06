#include <stdio.h>
#include <math.h>
#include <xmmintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Add timing support
#include <sys/time.h>
double time_stamp()
{
  struct timeval t;
  double time;
  gettimeofday(&t, (struct timezone *)NULL);
  time = t.tv_sec + 1.0e-6*t.tv_usec;
  return time;
}
double time1, time2;

void driver(void);
void initialize(void);
void jacobi(void);
void jacobi_SIMD(void);
void error_check(void);

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

#define MSIZE 1026 
int n,m,mits; 
#define REAL float // flexible between float and double
REAL tol,relax=1.0,alpha=0.0543; 
REAL u[MSIZE][MSIZE],f[MSIZE][MSIZE],uold[MSIZE][MSIZE];
REAL dx,dy;

int main (void) 
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
  n=MSIZE;
  m=MSIZE;
  tol=0.0000000001;
  mits=5000;
#if 0 // Not yet support concurrent CPU and GPU threads  
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    printf("Running using %d threads...\n",omp_get_num_threads());
  }
#endif
#endif  
  driver ( ) ;
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

void driver( )
{
  initialize();

  time1 = time_stamp();
  /* Solve Helmholtz equation */
//  jacobi ();
  jacobi_SIMD();
  time2 = time_stamp();

  printf("------------------------\n");     
  printf("Execution time = %f\n",time2-time1);
  /* error_check (n,m,alpha,dx,dy,u,f)*/
  error_check ( );
}


/*      subroutine initialize (n,m,alpha,dx,dy,u,f) 
******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/

void initialize( )
{
      
      int i,j, xx,yy;
      //double PI=3.1415926;

      dx = 2.0 / (n-1);
      dy = 2.0 / (m-1);

/* Initialize initial condition and RHS */

//#pragma omp parallel for private(xx,yy,j,i)
       for (i=0;i<n;i++)
         for (j=0;j<m;j++)      
           {
            xx =(int)( -1.0 + dx * (i-1));        
            yy = (int)(-1.0 + dy * (j-1)) ;       
            u[i][j] = 0.0;                       
            f[i][j] = -1.0*alpha *(1.0-xx*xx)*(1.0-yy*yy)\
               - 2.0*(1.0-xx*xx)-2.0*(1.0-yy*yy);  
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

void jacobi( )
{
  REAL omega;
  int i,j,k;
  REAL error,resid,ax,ay,b;
  //      double  error_local;

  //      float ta,tb,tc,td,te,ta1,ta2,tb1,tb2,tc1,tc2,td1,td2;
  //      float te1,te2;
  //      float second;

  omega=relax;
  /*
   * Initialize coefficients */

  ax = 1.0/(dx*dx); /* X-direction coef */
  ay = 1.0/(dy*dy); /* Y-direction coef */
  b  = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */ 

  error = 10.0 * tol;
  k = 1;

// An optimization on top of naive coding: promoting data handling outside the while loop
// data properties may change since the scope is bigger:
#pragma omp target data map(in:n, m, omega, ax, ay, b, f[0:n][0:m]) map(inout:u[0:n][0:m]) map(alloc:uold[0:n][0:m])
  while ((k<=mits)&&(error>tol)) 
  {
    error = 0.0;    

    /* Copy new solution into old */
//#pragma omp parallel
//    {
#pragma omp target //map(in:n, m, u[0:n][0:m]) map(out:uold[0:n][0:m])
#pragma omp parallel for private(j,i)
      for(i=0;i<n;i++)   
        for(j=0;j<m;j++)
          uold[i][j] = u[i][j]; 

#pragma omp target //map(in:n, m, omega, ax, ay, b, f[0:n][0:m], uold[0:n][0:m]) map(out:u[0:n][0:m])
#pragma omp parallel for private(resid,j,i) reduction(+:error) // nowait  collapse(2) 
      for (i=1;i<(n-1);i++)  
        for (j=1;j<(m-1);j++)   
        { 
          resid = (ax*(uold[i-1][j] + uold[i+1][j])\
              + ay*(uold[i][j-1] + uold[i][j+1])+ b * uold[i][j] - f[i][j])/b;  

          u[i][j] = uold[i][j] - omega * resid;  
          error = error + resid*resid ;   
        }

//    }
    /*  omp end parallel */

    /* Error check */

    if (k%500==0)
      printf("Finished %d iteration with error =%f\n",k, error);
    error = sqrt(error)/(n*m);

    k = k + 1;
  }          /*  End iteration loop */

  printf("Total Number of Iterations:%d\n",k); 
  printf("Residual:%E\n", error); 

}

/*      subroutine error_check (n,m,alpha,dx,dy,u,f) 
      implicit none 
************************************************************
* Checks error between numerical and exact solution 
*
************************************************************/ 
void error_check ( )
{ 
  int i,j;
  REAL xx,yy,temp,error; 

  dx = 2.0 / (n-1);
  dy = 2.0 / (m-1);
  error = 0.0 ;

//#pragma omp parallel for private(xx,yy,temp,j,i) reduction(+:error)
  for (i=0;i<n;i++)
    for (j=0;j<m;j++)
    { 
      xx = -1.0 + dx * (i-1);
      yy = -1.0 + dy * (j-1);
      temp  = u[i][j] - (1.0-xx*xx)*(1.0-yy*yy);
      error = error + temp*temp; 
    }
  error = sqrt(error)/(n*m);
  printf("Solution Error :%E \n",error);
}


void jacobi_SIMD( )
{
  REAL omega;
  int i,j,k,iv;
  REAL error,resid,ax,ay,b;
  __m128 dl, dr, dt, db,d;
  __m128 ax_ps, ay_ps, b_ps;
  __m128 omega_ps, resid_ps, error_ps, f_ps;
  __m128 u_ps;
  //      double  error_local;

  //      float ta,tb,tc,td,te,ta1,ta2,tb1,tb2,tc1,tc2,td1,td2;
  //      float te1,te2;
  //      float second;

  omega=relax;
  /*
   * Initialize coefficients */

  ax = 1.0/(dx*dx); /* X-direction coef */
  ay = 1.0/(dy*dy); /* Y-direction coef */
  b  = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */ 

  error = 10.0 * tol;
  k = 1;

// An optimization on top of naive coding: promoting data handling outside the while loop
// data properties may change since the scope is bigger:
#pragma omp target data map(in:n, m, omega, ax, ay, b, f[0:n][0:m]) map(inout:u[0:n][0:m]) map(alloc:uold[0:n][0:m])
  while ((k<=mits)&&(error>tol)) 
  {
    error = 0.0;    
    error_ps = _mm_load1_ps(&error);

    /* Copy new solution into old */
//#pragma omp parallel
//    {
#pragma omp target //map(in:n, m, u[0:n][0:m]) map(out:uold[0:n][0:m])
#pragma omp parallel for private(j,i)
      for(i=0;i<n;i++)   
        for(j=0;j<m;j++)
          uold[i][j] = u[i][j]; 

      omega_ps = _mm_load1_ps(&omega);
      ax_ps = _mm_load1_ps(&ax);
      ay_ps = _mm_load1_ps(&ay);
      b_ps = _mm_load1_ps(&b);
#pragma omp target //map(in:n, m, omega, ax, ay, b, f[0:n][0:m], uold[0:n][0:m]) map(out:u[0:n][0:m])
#pragma omp parallel for private(resid,j,i,dl,dr,dt,db,f_ps,u_ps,error_ps) reduction(+:error) // nowait  collapse(2) 
      for (i=1;i<(n-1);i++) 
      { 
        for (j=1;j<(m-1);j+=4)   
        {
          d = _mm_loadu_ps((float*)(uold[i]+j)); 
          dt = _mm_loadu_ps((float*)(uold[i-1]+j)); 
          db = _mm_loadu_ps((float*)(uold[i+1]+j)); 
          dr = _mm_loadu_ps((float*)(uold[i]+j+1)); 
          dl = _mm_loadu_ps((float*)(uold[i]+j-1)); 
          f_ps = _mm_loadu_ps((float*)(f[i]+j)); 
//          resid = (ax*(uold[i-1][j] + uold[i+1][j])\
              + ay*(uold[i][j-1] + uold[i][j+1])+ b * uold[i][j] - f[i][j])/b;  
          resid_ps = _mm_mul_ps(ax_ps, _mm_add_ps(dt,db));
          resid_ps = _mm_add_ps(resid_ps,_mm_mul_ps(ay_ps, _mm_add_ps(dl,dr)));
          resid_ps = _mm_add_ps(resid_ps,_mm_mul_ps(b_ps, d));
          resid_ps = _mm_div_ps(_mm_sub_ps(resid_ps,f_ps),b_ps);

//          u[i][j] = uold[i][j] - omega * resid;  
//          error = error + resid*resid ;   
          u_ps = _mm_sub_ps(d, _mm_mul_ps(omega_ps, resid_ps));
          error_ps = _mm_add_ps(error_ps, _mm_mul_ps(resid_ps,resid_ps));
          _mm_storeu_ps((float*)(u[i]+j),u_ps);
//          printf("uold:\t %f %f %f %f\n",uold[i][j],uold[i][j+1],uold[i][j+1],uold[i][j+2],uold[i][j+3]);
//          printf("d:\t %f %f %f %f\n",*((float*)(&(d))),*((float*)(&d)+1),*((float*)(&d)+2),*((float*)(&d)+3));
//          printf("u:\t %f %f %f %f\n",u[i][j],u[i][j+1],u[i][j+2],u[i][j+3]);
//          printf("dnew:\t %f %f %f %f\n",*((float*)(&(u_ps))),*((float*)(&u_ps)+1),*((float*)(&u_ps)+2),*((float*)(&u_ps)+3));
        }
      }
      for(iv=0;iv<4;iv++)
            error = error + (*((float*)(&(error_ps))+iv)); 

//    }
    /*  omp end parallel */

    /* Error check */

    if (k%(500)==0)
    {
      printf("Finished %d iteration with error =%e\n",k, error);
    }
    error = sqrt(error)/(n*m);

    k = k + 1;
  }          /*  End iteration loop */
#if 0
      for (i=1;i<(n-1);i++) { 
        for (j=1;j<(m-1);j++)   
        {
           printf("%f ",u[i][j]);
        }
        printf("\n");
      }
#endif
  printf("Total Number of Iterations:%d\n",k); 
  printf("Residual:%E\n", error); 

}

