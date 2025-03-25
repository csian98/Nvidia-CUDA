/**
 * @file		main.cu
 * @brief		
 * @author		Jeong Hoon (Sian) Choi
 * @version 	1.0.0
 * @date		2024-04-03
 */

/* Copyright (C)
 * 2024 - Jeong Hoon (Sian) Choi
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <utility>
#include <algorithm>
#include <numeric>

#include "sample.cuh"

#include "sian/timer.h"

#if _TARGET_OS == OS_WINDOWS

#elif _TARGET_OS == OS_LINUX

// void custom_terminate_fnct(void) {
//	exit(1);
// }

template <typename T>
bool check_matrix(const T* a, const T* b, const int n, const int m) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int index = i * m + j;
			if (a[index] != b[index]) return false;
		}
	}
	return true;
}

template <typename T>
void single_thread(const T* a, const T* b, T* c, const int n, const int m, const int k) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int inner = 0;
			for (int c = 0; c < k; ++c) {
				inner += a[i * k + c] + b[c * m + j];
			}
			c[i * k + j] = inner;
		}
	}
}

template <typename T>
void multi_thread(const T* a, const T* b, T* c, const int n, const int m, const int k,
				  const int thread_index, const int thread_num) {
	int load = std::ceil(static_cast<float>(m) / thread_num);
	for (int i = 0; i < n; ++i) {
		for (int j = thread_index * load; j < (thread_index + 1) * load; ++j) {
			int inner = 0;
			for (int c = 0; c < k; ++c) {
				if (j > m) break;
				inner += a[i * k +c] + b[c * m + j];
			}
			c[i * k + j] = inner;
		}
	}
}

int main(int argc, char* argv[]) {
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	
	const int n = 1024;
	const int k = 1024;
	const int m = 2048;
	
	sian::Timer timer(3);


	std::cout << timer << std::cout;

	return 0;
}

#endif // OS dependency
