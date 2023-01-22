#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const char* blurHashForPixelsCUDA(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
	//cudaError_t cudaStatus;

	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) 
	//{
	//	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//	goto Error;
	//}

	float* factors = (float*)malloc(yComponents * xComponents * 3 * sizeof(float));
	if (!factors) return NULL;

	for (int yComponent = 0; yComponent < yComponents; yComponent++)
	{
		for (int xComponent = 0; xComponent < xComponents; xComponent++)
		{
			float r = 0, g = 0, b = 0;
			float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;

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
}