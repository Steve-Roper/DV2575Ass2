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
void FillHostMatrix(double** matrices, int size);
cudaError_t InitGPUData(double** matrices, int **dSize, int size, int **dStride, int stride);
cudaError_t TransferGPUData(double** matrices, int size, cudaMemcpyKind flag);

void ForwardElimination(double* matrix, int size);
void BackwardSubstitute(double* matrix, int size);

__global__ void ForwardEliminationColumn(double* matrix, int* size, int* row, int* stride);

int main()
{
	int stride			= 1;										//Number of columns handled by each thread
	int *dStride		= 0;
	dim3 grid_dim		= dim3(10, 1, 1);
	dim3 block_dim		= dim3(1024, 1, 1);

	double** matrices	= (double**)malloc(3 * sizeof(double*));	//0 CPU, 1 HGPU, 2 DGPU
	int size = 64;
	int *dSize = 0;

	for (size; size < 2049; size *= 2) //64, 128, 256, 512, 1024, 2048. Number of Rows/Columns, number of elements = size^2 + size
	{
		FILE *csv;
		csv = fopen("DV2575Ass2Times.csv", "a");
		fprintf(csv, "\nCPU,1,2,4,8\n");
		fclose(csv);
		double GPUTimes[50] = { 0.f };
		double CPUTimes[10] = { 0.f };
		for (int rep = 0; rep < 10; ++rep)
		{
			stride = 1;
			//Init matrices and variable storage
			InitCPUData(matrices, size);

			timespec before;
			timespec after;
			
			bool failed = false;
			for (stride; stride < 10; stride *= 2) //1, 2, 4, 8
			{
				//KERNEL CALL 1, Forward elimination
				int* dRow = 0;
				timespec_get(&before, TIME_UTC);
				cudaMalloc((void**)&dRow, sizeof(int));
				if (InitGPUData(matrices, &dSize, size, &dStride, stride) != cudaSuccess)
				{
					goto Error;
				}
				for (int i = 1; i < size; ++i)
				{
					cudaMemcpy(dRow, &i, sizeof(int), cudaMemcpyHostToDevice);
					ForwardEliminationColumn<<<grid_dim, block_dim>>>(matrices[2], dSize, dRow, dStride);
				}
				TransferGPUData(matrices, size, cudaMemcpyDeviceToHost);
				timespec_get(&after, TIME_UTC);
				double timeTakenSec = after.tv_sec - before.tv_sec;
				long long timeTakenNsec = after.tv_nsec - before.tv_nsec;
				long long timeTakenMsec = round(timeTakenNsec / 1000000.f);
				timeTakenSec += (double)timeTakenMsec / 1000.f;
				int timeArrayPos = (stride / 2) * 10 + rep;
				GPUTimes[timeArrayPos] = timeTakenSec;

				BackwardSubstitute(matrices[0], size);
				BackwardSubstitute(matrices[1], size);
		
				printf("Good result\n");
				/*for (int i = 1; i < (size + 1); ++i)
					printf("%f\t", matrices[1][i * size + i - 1]);*/
			Error:
				cudaFree(matrices[2]);
				cudaFree(dSize);
				cudaFree(dRow);
				cudaFree(dStride);

				FillHostMatrix(matrices, size);
			}

			timespec_get(&before, TIME_UTC);
			ForwardElimination(matrices[0], size);
			timespec_get(&after, TIME_UTC);
			double timeTakenSec = after.tv_sec - before.tv_sec;
			long long timeTakenNsec = after.tv_nsec - before.tv_nsec;
			long long timeTakenMsec = round(timeTakenNsec / 1000000.f);
			timeTakenSec += (double)timeTakenMsec / 1000.f;
			CPUTimes[rep] = timeTakenSec;
		}


		free(matrices[0]);
		free(matrices[1]);
		/*printf("Writing size %d to DV2575Ass2Times.csv\n", size);
		csv = fopen("DV2575Ass2Times.csv", "a");
		for (int j = 0; j < 10; ++j)
		{
			fprintf(csv, "%f,", CPUTimes[j]);
			for (int i = 0; i < 5; ++i)
			{
				if (i == 3)
				{
					++i;
				}
				fprintf(csv, "%f,", GPUTimes[i * 10 + j]);
			}
			fprintf(csv, "\n");
		}
		fclose(csv);*/
	}
	free(matrices);
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
			matrices[0][i * (size + 1) + j] = matrices[1][i * (size + 1) + j] = (double)(rand() % 10 + 1); //not allowing zeros b/c easier
		}
	}
}

void FillHostMatrix(double** matrices, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			matrices[1][i * (size + 1) + j] = matrices[0][i * (size + 1) + j];
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
				matrix[i * (_size + 1) + column + j] -= (ratio * matrix[(_row - 1) * (_size + 1) + column + j]);
				__syncthreads();
			}
		}
	}
}