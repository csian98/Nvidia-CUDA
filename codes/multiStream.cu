#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM_BLOCK		(128*1024)
#define ARRAY_SIZE	(1024*NUM_BLOCK)
#define NUM_STREAMS	4
#define WORK_LOAD		256

__global__ void kernel(int* _in, int* _out) {
	int tID=blockIdx.x*blockDim.x+threadIdx.x;

	int temp=0;
	int in=_in[tID];

	for(int i=0; i<WORK_LOAD; ++i) {
		temp=(temp+in*5)%10;
	}
	_out[tID]=temp;
}

int main(void) {
	int *in=NULL, *out=NULL, *dIn=NULL, *dOut=NULL;

	cudaMallocHost(&in, sizeof(int)*ARRAY_SIZE);
	cudaMallocHost(&out, sizeof(int)*ARRAY_SIZE);
	memset(in, 0, sizeof(int)*ARRAY_SIZE);
	memset(out, 0, sizeof(int)*ARRAY_SIZE);

	cudaMalloc(&dIn, sizeof(int)*ARRAY_SIZE);
	cudaMalloc(&dOut, sizeof(int)*ARRAY_SIZE);
	for(int i=0; i<ARRAY_SIZE; ++i) in[i]=rand()%10;

	cudaStream_t stream[NUM_STREAMS];
	for(int i=0; i<NUM_STREAMS; ++i) cudaStreamCreate(&stream[i]);

	int chunkSize=ARRAY_SIZE/NUM_STREAMS;

	for(int i=0; i<NUM_STREAMS; ++i) {
		int offset=chunkSize*i;
		cudaMemcpyAsync(dIn+offset, in+offset, sizeof(int)*chunkSize, cudaMemcpyHostToDevice, stream[i]);
	}

	for(int i=0; i<NUM_STREAMS; ++i) {
		int offset=chunkSize*i;
		kernel<<<NUM_BLOCK/NUM_STREAMS, 1024, 0, stream[i]>>>(dIn+offset, dOut+offset);
	}

	for(int i=0; i<NUM_STREAMS; ++i) {
		int offset=chunkSize*i;
		cudaMemcpyAsync(out+offset, dOut+offset, sizeof(int)*chunkSize, cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaDeviceSynchronize();

	for(int i=0; i<NUM_STREAMS; ++i) cudaStreamDestroy(stream[i]);

	cudaFree(dIn); cudaFree(dOut);
	cudaFreeHost(in); cudaFreeHost(out);

	return 0;
}
