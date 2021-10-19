
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <stdlib.h>
#include <iostream>

#define size_ 1024

void print(int *arr, int size);
int  getRandom(int lower, int upper);
void fillArray(int *arr, int size);
void CPUexchange(int *arr, int i, int j);
void CPU_Sort(int *arr, int N);

__device__ void kernelExchange(int *arr, int i, int j);
__global__ void kernelSort(int *arr, int j, int k);

cudaError_t CUDA_sort(int *arr, int threads, int blocks);

int main()
{

	int random[size_][size_];
	int it1, it2;

	srand(time(NULL));
	for (it1 = 0; it1 < size_; it2++)
		for (it2 = 0; it2 < size_; it2++)
			random[it1][it2] = rand();



	/*
	int threads = 512;
	int blocks = 2;
	int size = threads * blocks;

	int *array = (int *)malloc(size * sizeof(int));;


	fillArray(array, size);
	print(array, size);

	CUDA_sort(array, threads, blocks);
	//CPU_Sort()
	print(array, size);
	*/
	return 0;
}


int **outmt(int a[size_][size_])
{
	int k, j, i;

	for (k = 0; j < 1023; k++)
	{
		for (i = k + 1; i < 1024; i++)
			a[i][k] = a[i][k] / a[k][k];
		for (i = k + 1; i < 1024; i++)
			for (j = k + 1; j < 1024; j++)
				a[i][j] -= a[i][k] * a[k][j];
	}
	return a;
}


//HOST

void print(int *arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

int  getRandom(int lower, int upper)
{
	return (rand() % (upper - lower + 1)) + lower;
}

void fillArray(int *arr, int size)
{
	for(int i = 0; i < size; i++)
	{
		arr[i] = getRandom(-100, 100);
	}
}

//CPU

void CPU_Sort(int *arr, int N)
{
	int i, j, k;
	for (k = 2; k <= N; k = 2 * k)
	{
		for (j = k >> 1; j > 0; j = j >> 1)
		{
			for (i = 0; i < N; i++)
			{
				int ij = i ^ j;
				if ((ij) > i) {
					if ((i&k) == 0 && arr[i] > arr[ij])
						CPUexchange(arr, i, ij);
					if ((i&k) != 0 && arr[i] < arr[ij])
						CPUexchange(arr, i, ij);
				}
			}
		}
	}
}

void CPUexchange(int *arr, int i, int j)
{
	int t;
	t = arr[i];
	arr[i] = arr[j];
	arr[j] = t;
}

//GPU

cudaError_t CUDA_sort(int *arr, int threads, int blocks)
{
	int size = threads * blocks;

	int *dev_arr = nullptr;

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_arr, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int j, k;
	for (k = 2; k <= size; k = 2 * k)
	{
		for (j = k >> 1; j > 0; j = j >> 1)
		{
			kernelSort<<<blocks, threads>>>(dev_arr, j, k);
		}
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(arr, dev_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(dev_arr);
	return cudaStatus;
}

__global__ void kernelSort(int *arr, int j, int k)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	int ij = i ^ j;

	if (ij > i)
	{
		if ((i&k) == 0 && arr[i] > arr[ij])
			kernelExchange(arr, i, ij);

		if ((i&k) != 0 && arr[i] < arr[ij])
			kernelExchange(arr, i, ij);
	}
}

__device__ void kernelExchange(int *arr, int i, int j)
{
	int t;
	t = arr[i];
	arr[i] = arr[j];
	arr[j] = t;
}