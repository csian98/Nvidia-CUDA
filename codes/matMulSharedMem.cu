#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE	16;

__global__ void MatMulSharedMem(int* matA, int* matB, int* matC, int m, int n, int k) {
	int row=blockDim.x*blockIdx.x+threadIdx.x;
	int col=blockDim.y*blockIdx.y+threadIdx.y;

	int val=0;
	__shared__ int subA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int subB[BLOCK_SIZE][BLOCK_SIZE];

	int localRow=threadIdx.x;
	int localCol=threadIdx.y;

	for(int bID=0; bID<ceil((float)k/BLOCK_SIZE); bID++) {
		int stride=bID*BLOCK_SIZE;

		if(row>=m || stride+localCol>=k)
			subA[localRow][localCol]=0;
		else
			subA[localRow][localCol]=matA[row*k+(stride+localCol)];

		if(col>=n || stride+localRow>=k)
			subB[localRow][localCol]=0;
		else
			subB[localRow][localCol]=matB[(stride+localRow)*n+col];

		__syncthreads();

		for(int i=0; i<BLOCK_SIZE; ++i) {
			val+=subA[localRow][i]*subB[i][localCol];
		}

		__syncthreads();
	}
	if(row>=m || col>=n) return;

	matC[row*n+col]=val;
}