
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define uint unsigned int

void InitCPUData(double** matrix, int size);
cudaError_t InitGPUData(double** matrix, int *dSize ,int size);
cudaError_t TransferGPUData(double** matrix, int size, cudaMemcpyKind flag);

void ForwardElimination(double* matrix, int size);
void BackwardSubstitute(double* matrix, int size);

__global__ void ForwardElimination(double** matrix, int* size);
__global__ void BackwardSubstitute();

int main()
{
	dim3 grid_dim = dim3(1, 1, 1);
	dim3 block_dim = dim3(1024, 1, 1);
	
	const int size		= 4;										//Number of Rows/Columns, number of elements = size^2 + size
	int *dSize			= 0;
	double** matrices	= (double**)malloc(3 * sizeof(double*));	//0 CPU, 1 HGPU, 2 DGPU

	//Init matrices and variable storage
	InitCPUData(matrices, size);
	if (InitGPUData(matrices, dSize, size) != cudaSuccess)
	{
		goto Error;
	}
	printf("Initial matrix\n");
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[0][i * (size + 1) + j]);
		}
		printf("\n");
	}
	ForwardElimination(matrices[0], size);
	//KERNEL CALL 1, Forward elimination
	ForwardElimination<<<grid_dim, block_dim>>>(&matrices[0], dSize);
	TransferGPUData(matrices, size, cudaMemcpyDeviceToHost);
	printf("\n\nPost forward elimination\n");
	printf("CPU:\n");
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[0][i * (size + 1) + j]);
		}
		printf("\n");
	}
	printf("\nGPU:\n");
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[1][i * (size + 1) + j]);
		}
		printf("\n");
	}
	BackwardSubstitute(matrices[0],  size);
	printf("\n\nPost backward substitution\n");
	printf("CPU:\n");
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[0][i * (size + 1) + j]);
		}
		printf("\n");
	}
	printf("\nGPU:\n");
	/*for (int i = 0; i < size; ++i)
	{
		printf("%f\n", matrices[0][i * (size + 1) + size]);
	}
	printf("\n");*/

Error:
	for (int i = 0; i < 2; ++i)
		free(matrices[i]);
	free(matrices);
	system("PAUSE");
    return 0;
}

void InitCPUData(double** matrix, int size)
{
	srand((uint)time(NULL));
	//malloc number of rows
	matrix[0] = (double*)malloc(size * (size + 1) * sizeof(double*));
	matrix[1] = (double*)malloc(size * (size + 1) * sizeof(double*));
	
	for (int i = 0; i < size; ++i)
	{
		//fill row
		for (int j = 0; j < (size + 1); ++j)
		{
			matrix[0][i * (size + 1) + j] = matrix[1][i * (size + 1 ) + j] = (double)(rand() % 10 + 1); //not allowing zeros b/c easier
		}
	}
}

cudaError_t InitGPUData(double** matrices, int *dSize, int size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(&matrices[2], size * (size + 1) * sizeof(double*));
	if (cudaStatus == cudaSuccess)
	{
		cudaStatus = TransferGPUData(matrices, size, cudaMemcpyHostToDevice);
		if (cudaStatus == cudaSuccess)
		{
			cudaStatus = cudaMalloc(&dSize, sizeof(int));
			if (cudaStatus == cudaSuccess)
			{
				cudaStatus = cudaMemcpy(dSize, &size, sizeof(int), cudaMemcpyHostToDevice); //maybe move this to TransferGPUData?
				if (cudaStatus != cudaSuccess)
				{
					printf("\nCould not copy size variable from host to device\n");
				}
			}
			else
			{
				printf("\nCould not allocate device memory for matrix size\n");
			}
		}
	}
	else
	{
		printf("\nCould not allocate device memory for matrix\n");
	}

	

	return cudaStatus;
}

cudaError_t TransferGPUData(double** matrix, int size, cudaMemcpyKind flag)
{
	cudaError_t cudaStatus;
	int to = (flag == 1) + 1, from = (flag == 2) + 1;
	cudaStatus = cudaMemcpy(matrix[to], matrix[from], size * (size + 1) * sizeof(double), flag); //on (flag == 2/1) + 1, this will return 1 or 2, depending on what flag is input, meaning I copy either to or from GPU
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCould not copy matrix from ");
		flag == 1 ? printf("host to device\n") : printf("device to host\n");
	}
	return cudaStatus;
}

void ForwardElimination(double* matrix, int size)
{
	for (int i = 1; i < size; ++i)
	{
		for (int j = i; j < size; ++j)
		{
			//Calculate ratio between rows, so one can be reduced to 0
			double ratio = matrix[j * (size + 1) + i - 1] / matrix[(i - 1) * (size + 1) + (i - 1)]; //(i - 1) * (size + 2)
			for (int k = 0; k < (size + 1); ++k)
			{
				matrix[j * (size + 1) + k] -= (ratio * matrix[(i - 1) * (size + 1) + k]);
			}
		}
	}
}

void BackwardSubstitute(double* matrix, int size)
{
	for (int i = (size - 1); i > 0; --i)
	{
		//variables here would usually be x,y,z etc. as in a1x + b1y + c1z = s1
		//												   a2x + b2y + c2z = s2
		//												   a3x + b3y + c3z = s3
		matrix[i * (size + 1) + size] = matrix[i * (size + 1) + size] / matrix[i * (size + 1) + i];
		for (int j = i - 1; j > -1; --j)
		{
			//Subtract from the rightmost element
			matrix[j * (size + 1) + size] -= matrix[j * (size + 1) + i] * matrix[i * (size + 1) + size];
			//Eliminate element above
			matrix[j * (size + 1) + i] = 0;
		}
		matrix[i * (size + 1) + i] = 1.f;
	}
	matrix[size] = matrix[size] / matrix[0];
	matrix[0] = 1.f;
}

__global__ void ForwardElimination(double** matrix, int* size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (!index)
	{
		printf("\n\nSomething's happening!\n\n");
	}
	//parallellize over k in ForwardElimination
	for (int i = 1; i < *size; ++i)
	{
		for (int j = i; j < *size; ++j)
		{
			//Calculate ratio between rows, so one can be reduced to 0
			double ratio = (double)(*matrix)[j * (*size + 1) + i - 1] / (double)(*matrix)[(i - 1) * (*size + 1) + i - 1];
			(*matrix)[j * (*size + 1) + index] -= (ratio * (*matrix)[(i - 1) * (*size + 1) + index]);
		}
	}
}

