#ifndef __BLURHASH_ENCODE_H__
#define __BLURHASH_ENCODE_H__

#include <stdint.h>
#include <stdlib.h>
#include "common.h"
#include <string.h>

const char* blurHashForPixels(int xComponents, int yComponents, int width, int height, unsigned char* rgb, size_t bytesPerRow)
{
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

	return restOfTheAlgorithm(factors, xComponents, yComponents);
}

#endif
