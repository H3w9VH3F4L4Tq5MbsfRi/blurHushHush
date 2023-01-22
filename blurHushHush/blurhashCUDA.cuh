#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float sRGBToLinearCUDA(int value) 
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
	int colorOffset = yComponent * xComponents * width * height + xComponent * width * height + threadY * width + threadX;

	float basis = cosf(M_PI * xComponent * threadX / width) * cosf(M_PI * yComponent * threadY / height);

	dev_r[colorOffset] = basis * sRGBToLinearCUDA(dev_rgb[3 * threadX + 0 + threadY * bytesPerRow]);
	dev_g[colorOffset] = basis * sRGBToLinearCUDA(dev_rgb[3 * threadX + 1 + threadY * bytesPerRow]);
	dev_b[colorOffset] = basis * sRGBToLinearCUDA(dev_rgb[3 * threadX + 2 + threadY * bytesPerRow]);

	return;
}

const char* blurHashForPixelsCUDA(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	unsigned char *dev_rgb = 0;

	cudaStatus = cudaMalloc((void**)&dev_rgb, width * height * 3 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_rgb, rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	float *dev_factors = 0;

	cudaStatus = cudaMalloc((void**)&dev_factors, yComponents * xComponents * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	float *dev_r = 0;

	cudaStatus = cudaMalloc((void**)&dev_r, yComponents * xComponents * width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	float *dev_g = 0;

	cudaStatus = cudaMalloc((void**)&dev_g, yComponents * xComponents * width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	float *dev_b = 0;

	cudaStatus = cudaMalloc((void**)&dev_b, yComponents * xComponents * width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	KernelFill <<< yComponents * xComponents * height, width >> > (dev_r, dev_g, dev_b, dev_rgb, width, height, xComponents, yComponents, bytesPerRow);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//launch second cuda function

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//launch third cuda function

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	float* factors = (float*)malloc(yComponents * xComponents * 3 * sizeof(float));
	if (!factors) goto Error;

	cudaStatus = cudaMemcpy(factors, dev_factors, xComponents * yComponents * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



	for (int yComponent = 0; yComponent < yComponents; yComponent++)
	{
		for (int xComponent = 0; xComponent < xComponents; xComponent++)
		{
			float r = 0, g = 0, b = 0;
			
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					float basis = cosf(M_PI * xComponent * x / width) * cosf(M_PI * yComponent * y / height);
					r += basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
					g += basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
					b += basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);
				}
			}

			///////////////////////

			float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;
			float scale = normalisation / (width * height);

			static float result[3];
			result[0] = r * scale;
			result[1] = g * scale;
			result[2] = b * scale;

			float* factor = result;
			factors[(yComponent * xComponents + xComponent) * 3] = factor[0];
			factors[(yComponent * xComponents + xComponent) * 3 + 1] = factor[1];
			factors[(yComponent * xComponents + xComponent) * 3 + 2] = factor[2];
		}
	}

	return restOfTheAlgorithm(factors, xComponents, yComponents);

Error:
	cudaFree(dev_rgb);
	cudaFree(dev_factors);
	cudaFree(dev_r);
	cudaFree(dev_g);
	cudaFree(dev_b);

	return nullptr;
}