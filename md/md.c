/*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! This program implements a simple molecular dynamics simulation,
!   using the velocity Verlet time integration scheme. The particles
!   interact with a central pair potential.
!
! Author:   Bill Magro, Kuck and Associates, Inc. (KAI), 1998
!
! Parallelism is implemented via OpenMP directives.
! THIS PROGRAM USES THE FORTRAN90 RANDOM_NUMBER FUNCTION AND ARRAY 
!   SYNTAX
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
*/

void compute(int np,int nd,double* box,double* pos, double* vel, double mass, double* f, double pot, double kin);

int main()
{
  int ndim;       // dimensionality of the physical space
  int nparts;     // number of particles
  int nsteps;     // number of time steps in the simulation
  parameter(ndim=3,nparts=500,nsteps=1000)
  double mass;        // mass of the particles
  double dt;          // time step
  double box[ndim];   // dimensions of the simulation box
  const double mawss = 1.0;
  const double dt = 1.0e-4;
  //parameter(mass=1.0,dt=1.0e-4)

//  ! simulation variables
  double position[nparts][ndim];
  double velocity[nparts][ndim];
  double force[nparts][ndim];
  double accel[nparts][ndim];
  double potential, kinetic, E0;
  int i;

  for(int idx=0;idx<ndim;++idx)
    box[idx] = 10.;

    // set initial positions, velocities, and accelerations
    initialize(nparts,ndim,box,position,velocity,accel);

    // compute the forces and energies
    compute(nparts,ndim,box,position,velocity,mass,force,potential,kinetic);
    E0 = potential + kinetic;

    // This is the main time stepping loop
    for(i=1;i<=nsteps;++i)
    {
        compute(nparts,ndim,box,position,velocity,mass,force,potential,kinetic);
        printf("%f  %f  %f\n", potential, kinetic,(potential + kinetic - E0)/E0);
        update(nparts,ndim,position,velocity,force,accel,mass,dt);
    }

  return 0;
}
/*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute the forces and energies, given positions, masses,
! and velocities
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
*/
void compute(int np,int nd,double* box,double* pos, double* vel, double mass, double* f, double pot, double kin)
{
/*
      integer np
      integer nd
      real*8  box(nd)
      real*8  pos(nd,np)
      real*8  vel(nd,np)
      real*8  f(nd,np)
      real*8  mass
      real*8  pot
      real*8  kin
*/
/*
      real*8 dotr8
      external dotr8
*/
      double v, dv, x;

      int i, j, k;
      double  rij[nd];
      double  d;
      const double  PI2 = 3.14159265d0/2.0d0;
/*
      ! statement function for the pair potential and its derivative
      ! This potential is a harmonic well which smoothly saturates to a
      ! maximum value at PI/2.
*/
  
      v[x] = pow(sin(min(x,PI2)),2.);
      dv[x] = 2.*sin(min(x,PI2))*cos(min(x,PI2));

      pot = 0.0;
      kin = 0.0;

//      ! The computation of forces and energies is fully parallel.
//$omp  parallel do
//$omp& default(shared)
//$omp& private(i,j,k,rij,d)
//$omp& reduction(+ : pot, kin)
      for( i=1;i<np;++i)
//        ! compute potential energy and forces
        int idx;
        for(idx = 0; idx < nd; ++idx)
          f[i][idx] = 0.0;
        for(j=1;j<np;++j)
          {
             if (i != j)
	     { 
               dist(nd,box,pos(1,i),pos[j][1],rij,d);
//               ! attribute half of the potential energy to particle 'j'
               pot = pot + 0.5*v[d];
               for(k=1;k<nd;++k)
	       {
                 f[i][k] = f[i][k] - rij[k]*dv[d]/d;
               }
             }
        }
        ! compute kinetic energy
        kin = kin + dotr8(nd,vel[i][1],vel[i][1]);
      enddo
//!$omp  end parallel do
      kin = kin*0.5*mass;
  
      return;
}
/*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Initialize the positions, velocities, and accelerations.
! The Fortran90 random_number function is used to choose positions.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
*/
void initialize(int np,int nd,double* box,double* pos,double* vel,double* acc)
{
/*
integer np
integer nd
real*8  box(nd)
real*8  pos(nd,np)
real*8  vel(nd,np)
real*8  acc(nd,np)
*/
int i, j;
double x;

for(i=1;i<=np;++i)
{
  for(j=1;j<nd;++j)
  {
    random_number(x);
    pos[i][j] = box[j]*x;
    vel[i][j] = 0.0;
    acc[i][j] = 0.0;
  }
}

return;
}
//! Compute the displacement vector (and its norm) between two particles.
void dist(int nd,double* box,double* r1,double* r2,double* dr,double* d)
{
/*
integer nd
real*8 box(nd)
real*8 r1(nd)
real*8 r2(nd)
real*8 dr(nd)
real*8 d
*/
int i;

d = 0.0;
for(i=1;i<nd;++i)
{
  dr[i] = r1[i] - r2[i];
  d = d + pow(dr(i),2.);
}
d = sqrt(d);

return;
}

! Return the dot product between two vectors of type real*8 and length n
double function dotr8(int n,double* x,double* y)
{
/*
      integer n
      real*8 x(n)
      real*8 y(n)
*/
      int i;
      double dotr8;
      dotr8 = 0.0;
      for(i = 1;i<n;++i)
      {
        dotr8 = dotr8 + x[i]*y[i];
      }

      return dotr8;
}
/*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Perform the time integration, using a velocity Verlet algorithm
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
*/
subroutine update(int np,int nd,double* pos,double* vel,double* f,double* a,double mass,double dt)
{
/*
integer np
integer nd
real*8  pos(nd,np)
real*8  vel(nd,np)
real*8  f(nd,np)
real*8  a(nd,np)
real*8  mass
real*8  dt
*/
int i, j;
double  rmass;

rmass = 1.0/mass;

//! The time integration is fully parallel
//!$omp  parallel do
//!$omp& default(shared)
//!$omp& private(i,j)
for(i = 1;i<np;++i)
{
  for(j = 1;j<nd;++j)
  {
    pos[i][j] = pos[i][j] + vel[i][j]*dt + 0.5*dt*dt*a[i][j];
    vel[i][j] = vel[i][j] + 0.5*dt*(f[i][j]*rmass + a[i][j]);
    a[i][j] = f[i][j]*rmass;
  }
}
//!$omp  end parallel do
return;
}
