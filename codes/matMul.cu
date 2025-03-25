#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE	3

typedef struct {
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

__device__ float getElement(const Matrix A, int row, int col) {
	return A.elements[row*A.stride+col];
}

__device__ void setElement(Matrix A, int row, int col, float value) {
	A.elements[row*A.stride+col]=value;
}

__device__ Matrix getSubMatrix(Matrix A, int row, int col) {
	Matrix Asub;
	Asub.width=BLOCK_SIZE;
	Asub.height=BLOCK_SIZE;
	Asub.stride=A.stride;
	Asub.elements=&A.elements[A.stride*BLOCK_SIZE*row+BLOCK_SIZE*col];
	return Asub;
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

__host__ void MatMul(const Matrix A, const Matrix B, Matrix C) {
	Matrix d_A;
	d_A.width=d_A.stride=A.width;
	d_A.height=A.height;

	Matrix d_B;
	d_B.width=d_B.stride=B.width;
	d_B.height=B.height;

	Matrix d_C;
	d_C.width=d_C.stride=C.width;
	d_C.height=C.height;

	size_t size=A.width*A.height*sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	size=B.width*B.height*sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	size=C.width*C.height*sizeof(float);
	cudaMalloc(&d_C.elements, size);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
	int blockRow=blockIdx.x, blockCol=blockIdx.y;

	Matrix Csub=getSubMatrix(C, blockRow, blockCol);
	float Cvalue=0;

	int row=threadIdx.y, col=threadIdx.x;

	for(int m=0; m<(A.width/BLOCK_SIZE); ++m) {
		Matrix Asub=getSubMatrix(A, blockRow, m);
		Matrix Bsub=getSubMatrix(B, m, blockCol);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col]=getElement(Asub, row, col);
		Bs[row][col]=getElement(Bsub, row, col);

		__syncthreads();

		for(int e=0; e<BLOCK_SIZE; ++e) {
			Cvalue+=As[row][e]*Bs[e][col];
		}
		__syncthreads();
	}
	setElement(Csub, row, col, Cvalue);
}

int main(void) {
	Matrix A, B, C;

	A.width=9, A.height=9;
	B.width=9, B.height=9;
	C.width=9, C.height=9;

	A.elements=(float*)malloc(sizeof(float)*A.width*A.height);
	B.elements=(float*)malloc(sizeof(float)*B.width*B.height);
	C.elements=(float*)malloc(sizeof(float)*C.width*C.height);

	for(int i=0; i<A.width*A.height; ++i) {
		A.elements[i]=i; B.elements[i]=i;
	}
	
	MatMul(A, B, C);

	for(int i=0; i<C.width; ++i) {
		for(int j=0; j<C.height; ++j) {
			printf("%f ", C.elements[j*C.width+i]);
		}
		printf("\n");
	}
}