
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void InitMatrix(int** matrix, int size);
void Backropagate(int** matrix, int size);
void ForwardSubstitute(int** matrix, int size, float* variables);

int main()
{
	int** matrix;
	int size = 100;
	InitMatrix(matrix, size);
Error:
	for (int i = 0; i < size; ++i)
		free(matrix[i]);
	free(matrix);
    return 0;
}

void InitMatrix(int** matrix, int size)
{
	srand(time(NULL));
	matrix = (int**)malloc(size * sizeof(int*));
	for (int i = 0; i < size; ++i)
	{
		matrix[i] = (int*)malloc((size + 1) * sizeof(int));
		for (int j = 0; j < (size + 1); ++j)
		{
			matrix[i][j] = rand() % 10 + 1; //not allowing zeros b/c easier
		}
	}
}

void Backropagate(int** matrix, int size)
{
	for (int i = 1; i < size; ++i)
	{
		float ratio = (float)matrix[i][i - 1] / (float)matrix[i - 1][i - 1];
		for (int j = 0; j < size + 1; ++j)
		{
			int subtrahend = (int)round(ratio * matrix[i - 1][j]);
			matrix[i][j] -= subtrahend;
		}
	}
}

void ForwardSubstitute(int** matrix, int size, float* variables)
{
	for (int i = size; i > -1; --i)
	{
		//variables here would usually be x,y,z etc. as in a1x + b1y + c1z = s1
		//												   a2x + b2y + c2z = s2
		//												   a3x + b3y + c3z = s3
		variables[i] = (float)matrix[i][size + 1] / (float)matrix[i][i];
		matrix[i][size + 1] -= (int)round(matrix[i][i] * variables[i]);
		matrix[i][i] = 0;
	}
}