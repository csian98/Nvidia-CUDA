#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

__global__ void MatAdd_G2D_B2D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE) {
	unsigned int col=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int row=threadIdx.y+blockIdx.y*blockDim.y;
	unsigned int index=row*COL_SIZE+col;

	if(col<COL_SIZE && row<ROW_SIZE) MatC[index]=MatA[index]+MatB[index];
}

__global__ void MatAdd_G1D_B1D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE) {
	unsigned int col=threadIdx.x+blockIdx.x*blockDim.x;
	if(col<COL_SIZE) {
		for(int row=0; row<ROW_SIZE; row++) {
			int index=row*COL_SIZE+col;
			MatC[index]=MatA[index]+MatB[index];
		}
	}
}

__global__ void MatAdd_G2D_B1D(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE) {
	unsigned int col=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int row=blockIdx.y;

	unsigned index=row*COL_SIZE+col;

	if(col<COL_SIZE && row<ROW_SIZE) MatC[index]=MatA[index]+MatB[index];
}