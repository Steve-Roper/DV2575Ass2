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
cudaError_t CudaMemcpyMatrix(double** matrices, int size, cudaMemcpyKind flag);

void ForwardElimination(double* matrix, int size);
void BackwardSubstitute(double* matrix, int size);

__global__ void ForwardEliminationColumn(double* matrix, int* size, int* row, int* stride, int* pivotRow);

int main()
{
	dim3 grid_dim		= dim3(1, 1, 1);
	dim3 block_dim		= dim3(1024, 1, 1);

	double** matrices	= (double**)malloc(4 * sizeof(double*));		//0 CPU, 1 HGPU, 2 DGPU, 3 Backup
	int size			= 64;											//Number of Rows / Columns, number of elements = size ^ 2 + size
	int *dSize			= 0;

	for (size; size < 2049; size *= 2) //64, 128, 256, 512, 1024, 2048. 
	{
		int failed = 0;

		FILE *csv;
		csv = fopen("DV2575Ass2Times.csv", "a");
		fprintf(csv, "\nCPU,1,2,4,8\n");
		fclose(csv);
		double GPUTimes[50] = { 0.f };
		double CPUTimes[10] = { 0.f };
		for (int rep = 0; rep < 10; ++rep)
		{
			//Init matrices and variable storage
			InitCPUData(matrices, size);

			timespec before;
			timespec after;

			timespec_get(&before, TIME_UTC);
			ForwardElimination(matrices[0], size);
			timespec_get(&after, TIME_UTC);
			double timeTakenSec = after.tv_sec - before.tv_sec;
			long long timeTakenNsec = after.tv_nsec - before.tv_nsec;
			long long timeTakenMsec = round(timeTakenNsec / 1000000.f);
			timeTakenSec += (double)timeTakenMsec / 1000.f;
			CPUTimes[rep] = timeTakenSec;

			BackwardSubstitute(matrices[0], size);

			for (int stride = 1; stride < 9; stride *= 2) //1, 2, 4, 8
			{
				int *dStride = 0;
				int totalStride = stride * (size / (grid_dim.x * block_dim.x * stride/*Total number of threads, multiplied by the stride*/) + 1);
				//KERNEL CALL 1, Forward elimination
				int* dRow = 0;
				int* dPivotRow = 0;
				timespec_get(&before, TIME_UTC);
				cudaMalloc((void**)&dRow, sizeof(int));
				cudaMalloc((void**)&dPivotRow, sizeof(int));
				cudaError_t cudaStatus = InitGPUData(matrices, &dSize, size, &dStride, totalStride);
				if (cudaStatus != cudaSuccess)
				{
					goto Error;
				}
				for (int i = 0; i < size; ++i)
				{
					for (int j = i + 1; j < (size + 1); ++j)
					{
						cudaMemcpy(dPivotRow, &i, sizeof(int), cudaMemcpyHostToDevice);
						cudaMemcpy(dRow, &j, sizeof(int), cudaMemcpyHostToDevice);
						ForwardEliminationColumn << <grid_dim, block_dim >> >(matrices[2], dSize, dRow, dStride, dPivotRow);
					}
				}
				CudaMemcpyMatrix(matrices, size, cudaMemcpyDeviceToHost);
				
				timespec_get(&after, TIME_UTC);
				double timeTakenSec = after.tv_sec - before.tv_sec;
				long long timeTakenNsec = after.tv_nsec - before.tv_nsec;
				long long timeTakenMsec = round(timeTakenNsec / 1000000.f);
				timeTakenSec += (double)timeTakenMsec / 1000.f;
				int timeArrayPos = (stride / 2) * 10 + rep;
				GPUTimes[timeArrayPos] = timeTakenSec;

				BackwardSubstitute(matrices[1], size);

				for (int i = 0; i < size; ++i)
				{
					for (int j = 0; j < (size + 1); ++j)
					{
						if (matrices[1][i * (size + 1) + j] != matrices[0][i * (size + 1) + j])
						{
							failed = i * (size + 1) + j;
							break;
						}
					}
					if (failed)
						break;
				}

				if (failed)
				{
					printf("Bad result\n");
					printf("CPU:%f\t\t-\tGPU:%f\n", matrices[0][failed], matrices[1][failed]);
				}

			Error:
				cudaFree(matrices[2]);
				cudaFree(dSize);
				cudaFree(dRow);
				cudaFree(dStride);
				FillHostMatrix(matrices, size);
			}
		}


		free(matrices[0]);
		free(matrices[1]);
		/*if (!failed)
		{
			printf("Writing size %d to DV2575Ass2Times.csv\n", size);
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
			fclose(csv);
		}*/
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
	matrices[3] = (double*)malloc(size * (size + 1) * sizeof(double*));

	double *s = (double*)malloc(size * sizeof(double));

	for (int i = 0; i < size; ++i)
	{
		//fill row
		for (int j = 0; j < size; ++j)
		{
			matrices[0][i * (size + 1) + j] = matrices[1][i * (size + 1) + j] = matrices[3][i * (size + 1) + j] = (double)(rand() % 10 + 1); //not allowing zeros b/c easier
		}

		s[i] = (double)(rand() % 10 + 1);

		matrices[0][i * (size + 1) + j] = matrices[1][i * (size + 1) + j] = matrices[3][i * (size + 1) + j] = 1;
	}

	//Filling last column like this to ensure the system is solvable
	for (int i = 0; i < size; ++i)
	{
		for(int j = 0; j < size; ++j)
		{
			matrices[0][i * (size + 1) + size] = matrices[1][i * (size + 1) + size] = matrices[3][i * (size + 1) + size] += (s[j] * matrices[0][j]);
		}
	}
}

void FillHostMatrix(double** matrices, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			matrices[1][i * (size + 1) + j] = matrices[3][i * (size + 1) + j];
		}
	}
}

cudaError_t InitGPUData(double** matrices, int **dSize, int size, int **dStride, int stride)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&matrices[2], size * (size + 1) * sizeof(double*));
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCould not allocate device memory for matrix\n");
		return cudaStatus;
	}

	cudaStatus = CudaMemcpyMatrix(matrices, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)dSize, sizeof(int)); //double void pointer super imoprtant
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCould not allocate device memory for matrix size\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy((void*)*dSize, &size, sizeof(int), cudaMemcpyHostToDevice); //maybe move this to TransferGPUData?
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCould not copy size variable from host to device\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)dStride, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCould not allocate device memory for thread stride\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy((void*)*dStride, &stride, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCould not copy stride variable from host to device\n");
		return cudaStatus;
	}

	return cudaStatus;
}

cudaError_t CudaMemcpyMatrix(double** matrices, int size, cudaMemcpyKind flag)
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
			double ratio = matrix[j * (size + 1) + i - 1] / matrix[(i - 1) * (size + 1) + (i - 1)];
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

__global__ void ForwardEliminationColumn(double* matrix, int* size, int* row, int* stride, int* pivotRow)
{
	int _size			= *size;
	int _row			= *row;
	int _stride			= *stride;
	int _pivotRow		= *pivotRow;
	int startColumn		= (blockIdx.x * blockDim.x + threadIdx.x) * _stride;

	double pivot		= (double)matrix[_pivotRow * (_size + 1) + _pivotRow];
	double belowPivot	= (double)matrix[_row * (_size + 1) + _pivotRow];

	double ratio		= belowPivot / pivot;

	for (int i = 0; i < _stride; ++i)
	{
		if (startColumn + i < (_size + 1))
		{
			matrix[_row * (_size + 1) + startColumn + i] -= (ratio * matrix[_pivotRow * (_size + 1) + startColumn + i]);
			__syncthreads();
		}
	}
}