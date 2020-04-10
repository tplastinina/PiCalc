#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
// thrust library
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "ctime"


__global__ void calcPi(float *x, float *y, int *blocks_counts, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	__shared__ int counts[512];
	int count = 0;
	for (int i = idx; i < n; i += offset) {
		if (x[i] * x[i] + y[i] * y[i] < 1.0f) {
			count++;
		}
	}
	counts[threadIdx.x] = count;

	__syncthreads();

	if (threadIdx.x == 0) {
		int total = 0;
		for (int j = 0; j < 512; j++) {
			total += counts[j];
		}
		blocks_counts[blockIdx.x] = total;
	}
}


void gpu_fillRand(float *a, float *b, unsigned int size) {
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	curandGenerateUniform(prng, a, size);
	curandGenerateUniform(prng, b, size);
}

float calcPICPU(float *x, float *y, unsigned size){
	int count = 0;
	for (int i = 0; i < size; i++) {
		if (x[i] * x[i] + y[i] * y[i] < 1.0f) {
			count++;
		}
	}
	return float(count) * 4.0 / size;
}

int main()
{
	long n = 1024* 1024 * 16 * 2;

	float *hostX, *hostY, *devX, *devY;


	hostX = (float *)calloc(n, sizeof(float));
	hostY = (float *)calloc(n, sizeof(float));

	cudaMalloc((void **)&devX, n * sizeof(float));
	cudaMalloc((void **)&devY, n * sizeof(float));


	float* ptrX = thrust::raw_pointer_cast(&devX[0]);
	float* ptrY = thrust::raw_pointer_cast(&devY[0]);


	gpu_fillRand(ptrX, ptrY, n);

	cudaMemcpy(hostX, ptrX, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostY, ptrY, n * sizeof(float), cudaMemcpyDeviceToHost);

	clock_t startCPU;
	startCPU = clock();
	float cpu_result = calcPICPU(hostX, hostY, n);
	printf("\nCPU's time for pi: %f", (clock() - startCPU) / (double)CLOCKS_PER_SEC);
	printf("\nCPU's result: %f", cpu_result);


	int *dev_blocks_counts = 0, *blocks_counts = 0;
	float gpuTime = 0.0f;
	cudaEvent_t start, stop;

	int blockDim = 512;
	int gridDim = n / (128 * blockDim);

	blocks_counts = (int *)calloc(gridDim, sizeof(int));

	cudaMalloc((void **)&dev_blocks_counts, 512 * sizeof(int));
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemset(dev_blocks_counts, 0, sizeof(int));

	calcPi<<<gridDim, blockDim>>>(devX, devY, dev_blocks_counts, n);

	cudaMemcpy(blocks_counts, dev_blocks_counts, gridDim * sizeof(int), cudaMemcpyDeviceToHost);
	int count = 0;
	for (int i = 0; i < gridDim; i++) {
		count += blocks_counts[i];
	};

	float gpu_result = float(count) * 4 / float(n);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("\nGPU's time spent executing %s: %f seconds", "kernel", gpuTime / 1000);
	printf("\nGPU's result: %f\n", gpu_result);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(ptrX);
	cudaFree(ptrY);
	cudaFree(dev_blocks_counts);
	return 0;
}
