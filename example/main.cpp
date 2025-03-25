#include "include/kernelCall.cuh"

int main(void) {
	kernelCall();
	printf("Host code running on CPU\n");
	cudaDeviceSynchronize();

	return 0;
}
