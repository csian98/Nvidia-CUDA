#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// extern __shared__
// __constant__

__global__ void kernel(void) {}

// cudaFuncCachePreferShared : shared memory 48KB
// cudaFuncCachePreferEqual	 : shared memory 32KM
// cudaFuncCachePreferL1		 : shared memory 16KB
// cudaFuncCachePreferNone	 : No preference
// cudaFuncCacheConfig(kernel, cudaFuncCachePreferNone)