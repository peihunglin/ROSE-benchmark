CUDA_INSTALL_PATH := /opt/cuda/4.2/cuda
CUDA_INCLUDES := -I. -I${CUDA_INSTALL_PATH}/include 
CUDA_LINK := -L${CUDA_INSTALL_PATH}/lib64 
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc -m64 -O3 -gencode arch=compute_20,code=sm_20 -Xptxas -v --maxrregcount 63 -Xcompiler -mcmodel=medium 
CC=mpicc
CFLAGS=
DEPS = 

all:	jacobi
main.o: main.c jacobi.o
	$(CC) -c -o main.o main.c

jacobi.o: jacobi-mpi.cu
	$(NVCC) -c -o jacobi.o jacobi-mpi.cu

jacobi: main.o jacobi.o 
	$(CC) -o jacobi main.o jacobi.o -lcudart ${CUDA_LINK}

clean:
	rm jacobi *.o 
