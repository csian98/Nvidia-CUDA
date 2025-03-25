#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE	1000000

__global__ void vecAdd(float* a, float* b, float* c, int size) {
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<size) c[i]=a[i]+b[i];
}

__host__ void vecAddCPU(float* a, float* b, float* c, int size) {
	for(int i=0; i<size; ++i) {
		c[i]=a[i]+b[i];
	}
}

__host__ int main(void) {
	clock_t start, end;
	int size=SIZE;

	float* h_a=(float*)malloc(sizeof(float)*size);
	float* h_b=(float*)malloc(sizeof(float)*size);
	float* h_c=(float*)malloc(sizeof(float)*size);

	for(int i=0; i<size; ++i) {
		h_a[i]=i;
		h_b[i]=i;
		h_c[i]=0;
	}

	start=clock();
	vecAddCPU(h_a, h_b, h_c, size);
	end=clock();

	printf("time cost for CPU: %f\n", (float)(end-start)/CLOCKS_PER_SEC);

	float* d_a;
	cudaMalloc(&d_a, sizeof(float)*size);
	float* d_b;
	cudaMalloc(&d_b, sizeof(float)*size);
	float* d_c;
	cudaMalloc(&d_c, sizeof(float)*size);

	cudaMemcpy(d_a, h_a, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float)*size, cudaMemcpyHostToDevice);

	int threadsPerBlock=256;
	int blocksPerGrid=(size+threadsPerBlock-1)/threadsPerBlock;

	start=clock();
	vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
	end=clock();

	printf("time cost for GPU: %f\n", (float)(end-start)/CLOCKS_PER_SEC);

	cudaMemcpy(h_c, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	
	return 0;
}