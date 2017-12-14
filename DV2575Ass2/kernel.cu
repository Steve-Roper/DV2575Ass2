#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define uint unsigned int

#ifdef __INTELLISENSE__
void __syncthreads();
#endif // __INTELLISENSE__

void InitCPUData(double** matrices, int size);
cudaError_t InitGPUData(double** matrices, int **dSize, int size, int **dStride, int stride);
cudaError_t TransferGPUData(double** matrices, int size, cudaMemcpyKind flag);

void ForwardElimination(double* matrix, int size);
void BackwardSubstitute(double* matrix, int size);

__global__ void ForwardEliminationColumn(double* matrix, int* size, int* row, int* stride);

int main()
{
	int stride			= 4;										//Number of columns handled by each thread
	int *dStride		= 0;
	dim3 grid_dim		= dim3(1, 1, 1);
	dim3 block_dim		= dim3(16, 1, 1);

	int size			= 10;										//Number of Rows/Columns, number of elements = size^2 + size
	int *dSize			= 0;
	double** matrices	= (double**)malloc(3 * sizeof(double*));	//0 CPU, 1 HGPU, 2 DGPU

	//Init matrices and variable storage
	InitCPUData(matrices, size);
	if (InitGPUData(matrices, &dSize, size, &dStride, stride) != cudaSuccess)
	{
		goto Error;
	}

	ForwardElimination(matrices[0], size);
	//KERNEL CALL 1, Forward elimination
	int* dRow = 0;
	cudaMalloc((void**)&dRow, sizeof(int));
	for (int i = 1; i < size; ++i)
	{
		cudaMemcpy(dRow, &i, sizeof(int), cudaMemcpyHostToDevice);
		ForwardEliminationColumn<<<grid_dim, block_dim>>>(matrices[2], dSize, dRow, dStride);
	}
	TransferGPUData(matrices, size, cudaMemcpyDeviceToHost);

	BackwardSubstitute(matrices[0], size);
	BackwardSubstitute(matrices[1], size);
	bool failed = false;

	for (int i = 0; i < size; ++i)
	{
		if (matrices[0][size] != matrices[1][size])
		{
			failed = true;
			break;
		}
	}
	if(failed)
		printf("Bad result\n");
	else
	{
		printf("Good result\n");
		/*for (int i = 1; i < (size + 1); ++i)
			printf("%f\t", matrices[1][i * size + i - 1]);*/
	}
	printf("\n");
Error:
	free(matrices[0]);
	free(matrices[1]);
	cudaFree(matrices[2]);
	free(matrices);
	cudaFree(dSize);
	cudaFree(dRow);
	system("PAUSE");
	return 0;
}

void InitCPUData(double** matrices, int size)
{
	srand((uint)time(NULL));
	//malloc number of rows
	matrices[0] = (double*)malloc(size * (size + 1) * sizeof(double*));
	matrices[1] = (double*)malloc(size * (size + 1) * sizeof(double*));

	for (int i = 0; i < size; ++i)
	{
		//fill row
		for (int j = 0; j < (size + 1); ++j)
		{
			matrices[0][i * (size + 1) + j] = matrices[1][i * (size + 1) + j] = (double)(rand() % 10 + 1); //not allowing zeros b/c easie
		}
	}
}

cudaError_t InitGPUData(double** matrices, int **dSize, int size, int **dStride, int stride)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&matrices[2], size * (size + 1) * sizeof(double*));
	if (cudaStatus == cudaSuccess)
	{
		cudaStatus = TransferGPUData(matrices, size, cudaMemcpyHostToDevice);
		if (cudaStatus == cudaSuccess)
		{
			cudaStatus = cudaMalloc((void**)dSize, sizeof(int)); //double void pointer super imoprtant
			if (cudaStatus == cudaSuccess)
			{
				cudaStatus = cudaMemcpy((void*)*dSize, &size, sizeof(int), cudaMemcpyHostToDevice); //maybe move this to TransferGPUData?
				if (cudaStatus == cudaSuccess)
				{
					cudaStatus = cudaMalloc((void**)dStride, sizeof(int));
					if (cudaStatus == cudaSuccess)
					{
						cudaStatus = cudaMemcpy((void*)*dStride, &stride, sizeof(int), cudaMemcpyHostToDevice);
						if (cudaStatus != cudaSuccess)
						{
							printf("\nCould not copy stride variable from host to device\n");
						}
					}
					else
					{
						printf("\nCould not allocate device memory for thread stride\n");
					}
				}
				else
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


Error:
	return cudaStatus;
}

cudaError_t TransferGPUData(double** matrices, int size, cudaMemcpyKind flag)
{
	cudaError_t cudaStatus;
	int to = (flag == 1) + 1, from = (flag == 2) + 1;
	cudaStatus = cudaMemcpy(matrices[to], matrices[from], size * (size + 1) * sizeof(double), flag);
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

__global__ void ForwardEliminationColumn(double* matrix, int* size, int* row, int* stride)
{
	int _size = *size;
	int _row = *row;
	int _stride = *stride;
	int column = (blockIdx.x * blockDim.x + threadIdx.x) * _stride;
	//parallellize over k in ForwardElimination
	for (int i = _row; i < (_size + 1); ++i)
	{
		double pivot = matrix[(_row - 1) * (_size + 1) + _row - 1];
		double thisElement = matrix[i * (_size + 1) + _row - 1];
		double ratio = thisElement / pivot;
		for (int j = 0; j < _stride; ++j)
		{
			if (column + j < (_size + 1))
			{
				//Calculate ratio between rows, so one can be reduced to 0
				//double ratio = (double)matrix[i * (_size + 1) + _row - 1] / (double)matrix[(_row - 1) * (_size + 1) + _row - 1 + j];
				matrix[i * (_size + 1) + column + j] -= (ratio * matrix[(_row - 1) * (_size + 1) + column + j]);
				__syncthreads();
			}
		}
	}
}