#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main(void) {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	//
	Kernel<<<dimGrid, dimBlock, 0, stream>>>();

	cudaStreamDestroy(stream);
}