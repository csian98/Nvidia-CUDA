#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM_STREAMS	4

__globa__ void kernel(int* in, int* out) {
	// workload;
}

int main(void) {
	cudaStream_t stream[NUM_STREAMS];
	cudaEvent_t start[NUM_STREAMS], end[NUM_STREAMS];

	for(int i=0; i<NUM_STREAMS; ++i) {
		cudaStreamCreate(&stream[i]);
		cudaEventCreate(&start[i]);
		cudaEventCreate(&end[i]);
	}
	//
	// omit
	//
	int chunkSize=ARRAY_SIZE/NUM_STREAMS;

	int offset[NUM_STREAMS]={0, };
	for(int i=0; i<NUM_STREAMS; ++i) offset[i]=chunkSize*i;

	for(int i=0; i<NUM_STREAMS; ++i) {
		cudaEventRecord(start[i], stream[i]);
		cudaMemcpyAsync(dIn+offset[i], in+offset[i], sizeof(int)*chunkSize, cudaMemcpyHostToDevice, stream[i]);
	}

	for(int i=0; i<NUM_STREAMS; ++i) {
		kernel<<<chunkSize/1024, 1024, 0, stream[i]>>>(dIn+offset[i], dOut+offset[i]);
	}

	for(int i=0; i<NUM_STREAMS; ++i) {
		cudaMemcpyAsync(out+offset[i], dOut+offset[i], sizeof(int)*chunkSize, cudaMemcpyDeviceToHost, stream[i]);
		cudaEventRecord(end[i], stream[i]);
	}

	cudaDeviceSynchronize();
	
	for(int i=0; i<NUM_STREAMS; ++i) {
		float time=0;
		cudaEventElapsedTime(&time, start[i], end[i]);
		printf("Stream[%d] : %f ms\n", i, time);
	}

	return 0;
}