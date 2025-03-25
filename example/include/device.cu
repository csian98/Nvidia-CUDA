#include "kernelCall.cuh"

__global__ void kernel(void) {
	printf("Device Code Running on the GPU\n");
}

void kernelCall(void) {
	kernel<<<1, 10>>>();
}