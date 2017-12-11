
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define uint unsigned int

cudaError_t InitMatrices(double** matrix, double** variables, int size);
void ForwardElimination(double* matrix, int size);
void BackwardSubstitute(double* matrix, double* variables, int size);
__global__ void Gaussian(double*** matrix, int* size);
__global__ void ForwardElimination();
__global__ void BackwardSubstitute();

int main()
{
	//Number of Rows/Columns
	const int size		= 4;
	//0 CPU, 1 HGPU, 2 DGPU
	double** matrices	= (double**)malloc(3 * sizeof(double*));
	double** variables	= (double**)malloc(3 * sizeof(double*));

	//Init matrices and variable storage
	if (InitMatrices(matrices, variables, size) != cudaSuccess)
	{
		goto Error;
	}
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[0][i * (size + 1) + j]);
		}
		printf("\n");
	}
	printf("\n");
	ForwardElimination(matrices[0], size);
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[0][i * (size + 1) + j]);
		}
		printf("\n");
	}
	printf("\n");
	BackwardSubstitute(matrices[0], variables[0], size);
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[0][i * (size + 1) + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < size; ++i)
	{
		printf("%f\n", variables[0][i]);
	}
	printf("\n");

	system("PAUSE");
Error:
	for (int i = 0; i < 2; ++i)
		free(matrices[i]);
	free(matrices);
	free(variables);
    return 0;
}

cudaError_t InitMatrices(double** matrix, double** variables, int size)
{
	srand((uint)time(NULL));
	//malloc number of rows
	matrix[0] = (double*)malloc(size * (size + 1) * sizeof(double*));
	matrix[1] = (double*)malloc(size * (size + 1) * sizeof(double*));
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(&matrix[2], size * (size + 1) * sizeof(double*));
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not allocate device memory for matrix\n");
	}
	for (int i = 0; i < size; ++i)
	{
		//fill row
		for (int j = 0; j < (size + 1); ++j)
		{
			matrix[0][i * (size + 1) + j] = matrix[1][i * (size + 1 ) + j] = (double)(rand() % 10 + 1); //not allowing zeros b/c easier
		}
	}

	if (cudaStatus == cudaSuccess)
	{
		cudaStatus = cudaMemcpy(matrix[2], matrix[0], (size + 1) * sizeof(double), cudaMemcpyHostToDevice);
	}
	//malloc variables (x,y,z etc.)
	variables[0] = (double*)malloc(size * sizeof(double*));
	variables[1] = (double*)malloc(size * sizeof(double*));

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

void BackwardSubstitute(double* matrix, double* variables, int size)
{
	for (int i = (size - 1); i > 0; --i)
	{
		//variables here would usually be x,y,z etc. as in a1x + b1y + c1z = s1
		//												   a2x + b2y + c2z = s2
		//												   a3x + b3y + c3z = s3
		variables[i] = matrix[i * (size + 1) + size] / matrix[i * (size + 1) + i];
		for (int j = i - 1; j > -1; --j)
		{
			//Subtract from the rightmost element
			matrix[j * (size + 1) + size] -= matrix[j * (size + 1) + i] * variables[i];
			//Eliminate element above
			matrix[j * (size + 1) + i] = 0;
		}
		matrix[i * (size + 1) + i] = 1.f;
	}
	variables[0] = matrix[size] / matrix[0];
}

__global__ void ForwardElimination(double** matrix, int* size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//parallellize over k in ForwardElimination
	for (int i = 1; i < *size; ++i)
	{
		for (int j = i; j < *size; ++j)
		{
			//Calculate ratio between rows, so one can be reduced to 0
			double ratio = (double)(*matrix)[j * (*size + 1) * i - 1] / (double)(*matrix)[(i - 1) * (*size + 1) + i - 1];
			(*matrix)[j * (*size + 1) + index] -= (ratio * (*matrix)[(i - 1) * (*size + 1) + index]);
		}
	}
}

