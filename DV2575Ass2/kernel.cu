
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void InitMatrix(int** matrix, int size);

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
