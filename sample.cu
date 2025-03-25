
/**
 * @file		sample.cu
 * @brief		
 * @author		Jeong Hoon (Sian) Choi
 * @version 	1.0.0
 * @date		2024-04-03
 */

#include "sample.cuh"

/* C & CPP */
/*
#ifdef __cplusplus
extern "C" {
#endif
*/

#if _TARGET_OS == OS_WINDOWS

#elif _TARGET_OS == OS_LINUX

/* Definition */

/* Functions definition */
__device__ void device(void) {
	
}

__global__ void kernel(void) {
	device();
}

#endif // OS dependency

/*
#ifdef __cplusplus
}
#endif
*/
