#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const char* blurHashForPixelsCUDA(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	if (xComponents < 1 || xComponents > 9) return NULL;
	if (yComponents < 1 || yComponents > 9) return NULL;

	int three = 3;

	float* factors = (float*)malloc(yComponents * xComponents * 3 * sizeof(float));
	if (!factors) return NULL;

	for (int y = 0; y < yComponents; y++)
	{
		for (int x = 0; x < xComponents; x++)
		{
			float* factor = multiplyBasisFunction(x, y, width, height, rgb, bytesPerRow);
			factors[(y * xComponents + x) * 3] = factor[0];
			factors[(y * xComponents + x) * 3 + 1] = factor[1];
			factors[(y * xComponents + x) * 3 + 2] = factor[2];
		}
	}

	float* dc = factors;
	float* ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char* ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if (acCount > 0)
	{
		float actualMaximumValue = 0;
		for (int i = 0; i < acCount * 3; i++)
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	}
	else
	{
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for (int i = 0; i < acCount; i++)
	{
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;

	return buffer;
}