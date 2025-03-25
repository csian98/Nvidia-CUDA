/**
 * @File		main.cu
 * @brief		
 * @author		Jeong Hoon (Sian) Choi
 * @version 	1.0.0
 * @date		2024-05-19
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

#include <random>
#include <cmath>

#include <string_view>
#include <string>
#include <vector>

#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>

#include "sian/timer.h"

#if _TARGET_OS == OS_WINDOWS

#elif _TARGET_OS == OS_LINUX

// void custom_terminate_fnct(void) {
//	exit(1);
// }

template <typename T>
bool check_matrix(const T* a, const T* b, const int n, const int m, const double tolerance = 1e-5) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int index = i * m + j;
			if (std::fabs(a[index] - b[index]) > tolerance) {
				std::cout << a[index] << " : " << b[index] << "[" << index << "]" <<std::endl;
				return false;
			}
		}
	}
	return true;
}

template <typename T>
void single_thread(const T* a, const T* b, T* c, const int n, const int m, const int k) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			double value = 0;
			for (int l = 0; l < k; ++l) {
				value += a[i * k + l] * b[l * m + j];
			}
			c[i * m + j] += value;
		}
	}
}

template <typename T>
void multi_thread(const T* a, const T* b, T* c, const int n, const int m, const int k,
				  const int thread_index, const int thread_num) {
	const int tasks = std::ceil(static_cast<float>(m) / thread_num);
	for (int i = 0; i < n; ++i) {
		for (int j = thread_index * tasks; j < (thread_index + 1) * tasks; ++j) {
			double value = 0;
			for (int l = 0; l < k; ++l) {
				if (j < m)
					value += a[i * k + l] * b[l * m + j];
			}
			c[i * m + j] += value;
		}
	}
}

static const int block_size = 32;

template <typename T>
__global__ void cuda_kernel(const T* a, const T* b, T* c, const int n, const int m, const int k) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int local_col = threadIdx.x;
	int local_row = threadIdx.y;
	
	__shared__ T partial_a[block_size][block_size];
	__shared__ T partial_b[block_size][block_size];

	double value = 0;
	
	for (int blk = 0; blk < std::ceil(static_cast<float>(k) / block_size); ++blk) {
		int stride = blk * block_size;

	    if (row >= n || stride + local_col >= k)
			partial_a[local_row][local_col] = 0;
		else
			partial_a[local_row][local_col] = a[row * k + (stride + local_col)];

		if (col >= m || stride + local_row >= k)
			partial_b[local_col][local_row] = 0;	// transpose (bank-confilic minimize)
		else
			partial_b[local_col][local_row] = b[(stride + local_row) * m + col];	// transpose
			
		__syncthreads();

		for (int i = 0; i < block_size; ++i) {
			value += partial_a[local_row][i] * partial_b[local_col][i];	// partial_b transpose
		}
		__syncthreads();
	}
	if (row >= n || col >= m) return;
	
	c[m * row + col] = value;
}

int main(int argc, char* argv[]) {
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	sian::Timer timer(3);

	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	
	const int n = 1024 * 4;
	const int k = 1024 * 4;
	const int m = 1024 * 4;

	double* a = new double[n * k];
	double* b = new double[k * m];
	double* c1 = new double[n * m];
	double* c2 = new double[n * m];
	double* c3 = new double[n * m];

	for (int i = 0; i < n * k; ++i) a[i] = distribution(engine);
	for (int i = 0; i < k * m; ++i) b[i] = distribution(engine);

	memset(c1, 0, sizeof(n * m));
	memset(c2, 0, sizeof(n * m));

	std::cout << "####\nMatrix Multiply Parallel Calculation\n####\n" << std::endl;
	
	timer[0].set_name("single thread");
	timer[0].start();
	single_thread(a, b, c1, n, m ,k);
	timer[0].stop();
	
	timer[1].set_name("multi threads");
	const auto thread_num = std::thread::hardware_concurrency();
	std::vector<std::thread> threads;
	timer[1].start();
	for (int i = 0; i < thread_num; ++i) {
		threads.emplace_back(&multi_thread<double>, a, b, c2, n, m, k, i, thread_num);
	}

	for (auto& thread : threads) thread.join();
	timer[1].stop();
	std::cout << "multi thread is correct : " << std::boolalpha << check_matrix(c1, c2, n, m) << std::endl;

	timer[2].set_name("cuda GPU SIMT");
	timer[2].start();

	dim3 grid_dim(std::ceil(static_cast<float>(m) / block_size),
				  std::ceil(static_cast<float>(n) / block_size));
	dim3 block_dim(block_size, block_size);
	double* d_a;
	double* d_b;
	double* d_c;
	cudaMalloc(&d_a, sizeof(double) * n * k);
	cudaMalloc(&d_b, sizeof(double) * k * m);
	cudaMalloc(&d_c, sizeof(double) * n * m);

	cudaMemcpy(d_a, a, sizeof(double) * n * k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(double) * k * m, cudaMemcpyHostToDevice);
	cudaMemset(d_c, 0, sizeof(double) * n * k);

	cuda_kernel<double><<<grid_dim, block_dim>>>(d_a, d_b, d_c, n, m ,k);

	cudaMemcpy(c3, d_c, sizeof(double) * n * m, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaDeviceSynchronize();
	timer[2].stop();
	std::cout << "cuda GPU SIMT is correct : " << std::boolalpha << check_matrix(c1, c3, n, m) << std::endl;
	
	std::cout << timer;

	delete[] c2;
	delete[] c1;
	delete[] b;
	delete[] a;

	return 0;
}

#endif // OS dependency
