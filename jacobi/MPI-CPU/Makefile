CC=/nfs/apps/mpich2/default/bin/mpicc
CFLAGS= -O3 -lm -g

all:	jacobi
jacobi: jacobi-mpi.c 
	$(CC) $(CFLAGS)  -o jacobi jacobi-mpi.c

clean:
	rm jacobi *.o 
