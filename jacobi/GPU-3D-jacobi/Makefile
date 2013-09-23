CUDA_INSTALL_PATH := /opt/cuda/4.2/cuda
CUDA_INCLUDES := -I. -I${CUDA_INSTALL_PATH}/include 
CUDA_LINK := -L${CUDA_INSTALL_PATH}/lib64 -lcuda -lcudart -lgomp 
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc -m64 -O3 -gencode arch=compute_20,code=sm_20 -Xptxas -v --maxrregcount 63 -Xcompiler -mcmodel=medium 
CC=mpicc
CFLAGS=
DEPS = 

all:	heat3D

heat3D: mint_heat3D.cu
	$(NVCC) ${CUDA_LINK} -o heat3D mint_heat3D.cu

clean:
	rm heat3D *.o 
