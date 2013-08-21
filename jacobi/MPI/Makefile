CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_INCLUDES := -I. -I${CUDA_INSTALL_PATH}/include -I/usr/local/NVIDIA_GPU_Computing_SDK/C/common/inc/
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
	$(CC) -o jacobi main.o jacobi.o -lcudart -L/usr/local/cuda/lib64/

clean:
	rm jacobi *.o 
