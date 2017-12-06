
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void InitMatrix(float*** matrix, float** variables, int size);
void Backropagate(float** matrix, int size);
void ForwardSubstitute(float** matrix, float* variables, int size);

int main()
{
	float** matrix = 0;
	float* variables = 0;
	int size = 4;
	//CPU Gaussian elimination
	InitMatrix(&matrix, &variables, size);
	Backropagate(matrix, size);
	ForwardSubstitute(matrix, variables, size);
	for (int i = 0; i < size; ++i)
	{
		printf("%f\n", variables[i]);
	}
	printf("\n");

	//GPU Gaussian elimination

	system("PAUSE");
Error:
	for (int i = 0; i < size; ++i)
		free(matrix[i]);
	free(matrix);
	free(variables);
    return 0;
}

void InitMatrix(float*** matrix, float** variables, int size)
{
	srand(time(NULL));
	//malloc number of rows
	*matrix = (float**)malloc(size * sizeof(float*));
	for (int i = 0; i < size; ++i)
	{
		//malloc a row
		(*matrix)[i] = (float*)malloc((size + 1) * sizeof(float));
		//fill row
		for (int j = 0; j < (size + 1); ++j)
		{
			(*matrix)[i][j] = (float)(rand() % 10 + 1); //not allowing zeros b/c easier
		}
	}
	//malloc variables (x,y,z etc.)
	*variables = (float*)malloc(size * sizeof(float*));
}

void Backropagate(float** matrix, int size)
{
	for (int i = 1; i < size; ++i)
	{
		for (int j = i; j < size; ++j)
		{
			//Calculate ratio between rows, so one can be reduced to 0
			float ratio = (float)matrix[j][i - 1] / (float)matrix[i - 1][i - 1];
			for (int k = 0; k < (size + 1); ++k)
			{
				matrix[j][k] -= (ratio * matrix[i - 1][k]);
			}
		}
	}
}

void ForwardSubstitute(float** matrix, float* variables, int size)
{
	for (int i = (size - 1); i > 0; --i)
	{
		//variables here would usually be x,y,z etc. as in a1x + b1y + c1z = s1
		//												   a2x + b2y + c2z = s2
		//												   a3x + b3y + c3z = s3
		variables[i] = matrix[i][size] / matrix[i][i];
		for (int j = i - 1; j > -1; --j)
		{
			//Subtract from the rightmost element
			matrix[j][size] -= matrix[j][i] * variables[i];
			//Eliminate element above
			matrix[j][i] = 0;
		}
	}
	variables[0] = matrix[0][size] / matrix[0][0];
}