#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernel(int* in, int* out) {
	int deviceID;
	cudaGetDevice(&deviceID);
	//
}

int main(void) {
	int ngpus;
	cudaGetDeviceCount(&ngpus);

	for(int i=0; i<ngpus; ++i) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		//
	}

	for(int i=0; i<ngpus; ++i) {
		cudaSetDevice(i);
		//cudaMemcpy(...);
		//kernel0<<<...>>>(...);
		//cudaMemcpy(...);
	}
}