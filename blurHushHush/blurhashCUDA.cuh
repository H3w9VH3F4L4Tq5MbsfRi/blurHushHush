#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float sRGBToLinearOnCUDA(int value) 
{
	float v = (float)value / 255;
	if (v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
}

__global__ void KernelFill(float* dev_r, float* dev_g, float* dev_b, unsigned char* dev_rgb, int width, int height, int xComponents, int yComponents, size_t bytesPerRow)
{
	int yComponent = blockIdx.x / (xComponents * height);
	int xComponent = (blockIdx.x - yComponent * xComponents * height) / height;
	int threadY = blockIdx.x - yComponent * xComponents * height - xComponent * height;
	int threadX = threadIdx.x;
	int colorOffset = blockIdx.x * width + threadX;

	float basis = cosf(M_PI * xComponent * threadX / width) * cosf(M_PI * yComponent * threadY / height);

	dev_r[colorOffset] = basis * sRGBToLinearOnCUDA(dev_rgb[3 * threadX + 0 + threadY * bytesPerRow]);
	dev_g[colorOffset] = basis * sRGBToLinearOnCUDA(dev_rgb[3 * threadX + 1 + threadY * bytesPerRow]);
	dev_b[colorOffset] = basis * sRGBToLinearOnCUDA(dev_rgb[3 * threadX + 2 + threadY * bytesPerRow]);

	return;
}

__global__ void KernelAddRows(float* dev_r, float* dev_g, float* dev_b, int width)
{
	int threadX = threadIdx.x;
	int colorOffset = blockIdx.x * width + threadX;

	for (int offset = 1; offset <= width; offset *= 2) 
	{
		__syncthreads();

		if (threadX % (offset * 2) != 0) continue;
		if (threadX + offset >= width) continue;

		dev_r[colorOffset] += dev_r[colorOffset + offset];
		dev_g[colorOffset] += dev_g[colorOffset + offset];
		dev_b[colorOffset] += dev_b[colorOffset + offset];
	}

	return;
}

__global__ void KernelAddColumnsAndRestOfAlg(float* dev_r, float* dev_g, float* dev_b, float* factors, int width, int height, int xComponents, int yComponents)
{
	int threadY = threadIdx.x;
	int colorOffset = blockIdx.x * width * height + threadY * width;

	for (int offset = 1; offset < height; offset *= 2)
	{
		__syncthreads();

		if (threadY % (offset * 2) != 0) continue; 
		if (threadY + offset >= height) continue;

		dev_r[colorOffset] += dev_r[colorOffset + offset * width];
		dev_g[colorOffset] += dev_g[colorOffset + offset * width];
		dev_b[colorOffset] += dev_b[colorOffset + offset * width];
	}

	__syncthreads();

	int sumIndx = blockIdx.x;
	int offset = width * height * sumIndx;
	float normalisation = ((sumIndx == 0) ? 1 : 2);
	float scale = normalisation / (width * height);

	if (threadY == 0)
	{
		factors[sumIndx * 3 + 0] = dev_r[offset] * scale;
		factors[sumIndx * 3 + 1] = dev_g[offset] * scale;
		factors[sumIndx * 3 + 2] = dev_b[offset] * scale;
	}

	return;
}

const char* blurHashForPixelsCUDA(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	unsigned char *dev_rgb = 0;

	cudaStatus = cudaMalloc((void**)&dev_rgb, width * height * 3 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMalloc on dev_rgb failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_rgb, rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy on dev_rgb failed!");
		goto Error;
	}

	float *dev_r = 0;

	cudaStatus = cudaMalloc((void**)&dev_r, yComponents * xComponents * width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc on dev_r failed!");
		goto Error;
	}

	float *dev_g = 0;

	cudaStatus = cudaMalloc((void**)&dev_g, yComponents * xComponents * width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc on dev_g failed!");
		goto Error;
	}

	float *dev_b = 0;

	cudaStatus = cudaMalloc((void**)&dev_b, yComponents * xComponents * width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc on dev_b failed!");
		goto Error;
	}

	KernelFill <<< (yComponents * xComponents * height), width >>> (dev_r, dev_g, dev_b, dev_rgb, width, height, xComponents, yComponents, bytesPerRow);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "KernelFill launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching KernelFill!\n", cudaStatus);
		goto Error;
	}

	KernelAddRows <<< (yComponents * xComponents * height), width >>> (dev_r, dev_g, dev_b, width);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "KernelAddRows launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching KernelAddRows!\n", cudaStatus);
		goto Error;
	}

	float* dev_factors = 0;

	cudaStatus = cudaMalloc((void**)&dev_factors, yComponents * xComponents * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc on dev_factors failed!");
		goto Error;
	}

	KernelAddColumnsAndRestOfAlg <<< (yComponents * xComponents), height >>> (dev_r, dev_g, dev_b, dev_factors, width, height, xComponents, yComponents);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "KernelAddColumnsAndRestOfAlg launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching KernelAddColumnsAndRestOfAlg!\n", cudaStatus);
		goto Error;
	}

	float* factors = (float*)malloc(yComponents * xComponents * 3 * sizeof(float));
	if (!factors) goto Error;

	cudaStatus = cudaMemcpy(factors, dev_factors, xComponents * yComponents * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy on dev_factors failed!");
		goto Error;
	}

	return restOfTheAlgorithm(factors, xComponents, yComponents);

Error:
	cudaFree(dev_rgb);
	cudaFree(dev_r);
	cudaFree(dev_g);
	cudaFree(dev_b);
	cudaFree(dev_factors);

	return nullptr;
}