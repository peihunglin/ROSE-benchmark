#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define _debug 1

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

#define MSIZE 16 
#define HALOSIZE 1 
int n,m,mits; 
#define REAL float // flexible between float and double
REAL tol,relax=1.0,alpha=0.0543; 
REAL dx,dy;

void driver(int, int);
void jacobi(float (*u)[m], float (*f)[m], float (*uold)[m], int rank, int nprocs, int ibegin, int isize);
void initialize(float (*u)[m], float (*f)[m], float (*uold)[m], int rank, int nprocs, int ibegin, int isize);
void error_check(void);

int main( int argc, char* argv[] )
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
// MPI setup
  int rank, nprocs;
  MPI_Status status;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
#if 0 // Not yet support concurrent CPU and GPU threads  
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    printf("Running using %d threads...\n",omp_get_num_threads());
  }
#endif
#endif  
#if _debug
  printf("I am rank %d out of total %d ranks\n",rank, nprocs);
#endif  
  driver (rank, nprocs) ;
  MPI_Finalize();
  return 0;
}


void dumpFile(float (*data)[m], int rank, int m_size, int n_size, char* prefix)
{
    FILE *initfile;
    int idx, idy;
    char name[15];
    char buffer[5];
    sprintf(buffer,"%d",rank);
    strcpy(name, prefix);
    strcat(name, buffer);
    strcat(name, ".txt");
    initfile = fopen(name,"w+");
    for(idx = 0; idx < n_size; ++idx)
    {
     fprintf(initfile, "row id = %d\n",idx);
    for(idy = 0; idy < m_size; ++idy)
    {
     fprintf(initfile," %f ", data[idx][idy]);
    }
     fprintf(initfile, "\n",idx);
    }
    fclose(initfile);
}

/*      subroutine initialize (n,m,alpha,dx,dy,u,f) 
******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/

void initialize(float (*u)[m], float (*f)[m], float (*uold)[m], int rank, int nprocs, int ibegin, int isize)
{
      
      int i,j, xx,yy;
      int iend = ibegin + isize -1;
      //double PI=3.1415926;
      dx = 2.0 / (n-1);
      dy = 2.0 / (m-1);

/* Initialize initial condition and RHS */

//#pragma omp parallel for private(xx,yy,j,i)
       for (i=ibegin;i<=iend;i++)
         for (j=0;j<m;j++)      
           {
            xx =(int)( -1.0 + dx * (i-1));        
            yy = (int)(-1.0 + dy * (j-1)) ;      
            u[i-ibegin][j] = 0.0;                       
            f[i-ibegin][j] = -1.0*alpha *(1.0-xx*xx)*(1.0-yy*yy)\
               - 2.0*(1.0-xx*xx)-2.0*(1.0-yy*yy);  
          }

    dumpFile(f,rank,m, isize,"init_");
}

/*************************************************************
* Subroutine driver () 
* This is where the arrays are allocated and initialzed. 
*
* Working varaibles/arrays 
*     dx  - grid spacing in x direction 
*     dy  - grid spacing in y direction 
*************************************************************/

void driver(int rank, int nprocs )
{
  REAL (*u)[m], (*f)[m], (*uold)[m];
  int isize;
  isize = 1 + ((n-1)/nprocs);
  int ibegin = rank * isize;
  int iend;
  iend = (((ibegin+isize) < n) ? (ibegin+isize):n) - 1;
  isize = iend - ibegin + 1;
  u = malloc(sizeof(*u) * isize);
  f = malloc(sizeof(*u) * isize);
  uold = malloc(sizeof(*u) * isize);
  memset(u, 0, sizeof(u[0][0]) * m * isize);
  memset(f, 0, sizeof(f[0][0]) * m * isize);
  memset(uold, 0, sizeof(uold[0][0]) * m * isize);

#if _debug
  printf("Inside Driver: I am rank %d and I will allocate %d rows of data\n",rank, isize);
#endif
  initialize(u, f, uold, rank, nprocs, ibegin, isize);

  time1 = time_stamp();
  /* Solve Helmholtz equation */
  jacobi(u, f, uold, rank, nprocs, ibegin, isize);
  time2 = time_stamp();

  printf("------------------------\n");     
//  printf("Execution time = %f\n",time2-time1);
  /* error_check (n,m,alpha,dx,dy,u,f)*/
//  error_check ( );
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

void jacobi(float (*u)[m], float (*f)[m], float (*uold)[m], int rank, int nprocs, int ibegin, int isize)
{
  REAL omega;
  int i,j,k;
  int n_local;
  REAL error,resid,ax,ay,b;
  //      double  error_local;

  //      float ta,tb,tc,td,te,ta1,ta2,tb1,tb2,tc1,tc2,td1,td2;
  //      float te1,te2;
#if _debug
  printf("Inside jacobi: I am rank %d and I will allocate %d rows of data\n",rank, isize);
#endif
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
  while ((k<=mits)&&(error>tol)) 
  {
    error = 0.0;    

    /* Copy new solution into old */
//#pragma omp parallel
//    {
#pragma omp parallel for private(j,i)
      for(i=0;i<isize;i++)   
        for(j=0;j<m;j++)
        {
          uold[i][j] = u[i][j]; 
        }

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

///*      subroutine error_check (n,m,alpha,dx,dy,u,f) 
//      implicit none 
//************************************************************
//* Checks error between numerical and exact solution 
//*
//************************************************************/ 
//void error_check ( )
//{ 
//  int i,j;
//  REAL xx,yy,temp,error; 
//
//  dx = 2.0 / (n-1);
//  dy = 2.0 / (m-1);
//  error = 0.0 ;
//
////#pragma omp parallel for private(xx,yy,temp,j,i) reduction(+:error)
//  for (i=0;i<n;i++)
//    for (j=0;j<m;j++)
//    { 
//      xx = -1.0 + dx * (i-1);
//      yy = -1.0 + dy * (j-1);
//      temp  = u[i][j] - (1.0-xx*xx)*(1.0-yy*yy);
//      error = error + temp*temp; 
//    }
//  error = sqrt(error)/(n*m);
//  printf("Solution Error :%E \n",error);
//}
//
//
