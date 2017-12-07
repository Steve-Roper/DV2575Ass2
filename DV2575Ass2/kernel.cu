
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void InitMatrices(float*** matrix, float** variables, int size);
void Backropagate(float** matrix, int size);
void ForwardSubstitute(float** matrix, float* variables, int size);

int main()
{
	//Number of Rows/Columns
	const int size = 4;

	//CPU data
	float** cMatrix		= 0;
	float* cVariables	= 0;	
	
	//GPU data
	float** hMatrix		= 0;
	float** dMatrix		= 0;
	float* hVariables	= 0;
	float* dVariables	= 0;

	//0 CPU, 1 HGPU, 2 DGPU
	float*** matrices = (float***)malloc(3 * sizeof(float**));//[3] = { cMatrix, hMatrix, dMatrix };
	float** variables = (float**)malloc(3 * sizeof(float*));//[3] = { cVariables, hVariables, dVariables };

	//Init matrices and variable storage
	InitMatrices(matrices, variables, size);
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < (size + 1); ++j)
		{
			printf("%f\t", matrices[0][i][j]);
		}
		printf("\n");
	}
	printf("\n");
	Backropagate(matrices[0], size);
	ForwardSubstitute(matrices[0], variables[0], size);
	for (int i = 0; i < size; ++i)
	{
		printf("%f\n", variables[0][i]);
	}
	printf("\n");

	system("PAUSE");
Error:
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < 2; ++j)
			free(matrices[j][i]);
	for (int i = 0; i < 2; ++i)
		free(matrices[i]);
	free(matrices);
	free(variables);
    return 0;
}

void InitMatrices(float*** matrix, float** variables, int size)
{
	srand(time(NULL));
	//malloc number of rows
	matrix[0] = (float**)malloc(size * sizeof(float*));
	matrix[1] = (float**)malloc(size * sizeof(float*));
	for (int i = 0; i < size; ++i)
	{
		//malloc a row
		matrix[0][i] = (float*)malloc((size + 1) * sizeof(float));
		matrix[1][i] = (float*)malloc((size + 1) * sizeof(float));
		//fill row
		for (int j = 0; j < (size + 1); ++j)
		{
			matrix[0][i][j] = matrix[1][i][j] = (float)(rand() % 10 + 1); //not allowing zeros b/c easier
		}
	}
	//malloc variables (x,y,z etc.)
	variables[0] = (float*)malloc(size * sizeof(float*));
	variables[1] = (float*)malloc(size * sizeof(float*));
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