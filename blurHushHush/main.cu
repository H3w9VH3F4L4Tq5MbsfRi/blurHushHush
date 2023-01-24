#define STB_IMAGE_IMPLEMENTATION

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stb_image.h"
#include "encode.h"
#include "blurhashCUDA.cuh"

#include <stdio.h>
#include <iostream>
#include <chrono>

#define megaToNormal 1000000

int main(int argc, char** argv)
{
    std::cout << std::endl;
    std::cout << "Patryk Saj" << std::endl;
    std::cout << "GPU Project 2" << std::endl;
    std::cout << "Blurhash" << std::endl << std::endl;

    if (argc != 4 || atoi(argv[1]) < 1 || atoi(argv[2]) < 1 || atoi(argv[1]) > 9 || atoi(argv[2]) > 9)
    {
        std::cout << "Invalid parameters!" << std::endl;
        std::cout << "Terminating program..." << std::endl;
        return 1;
    }

    //image loading
    std::cout << "Loading image..." << std::endl;
    auto start0 = std::chrono::high_resolution_clock::now();
    int width, height;

    unsigned char* data = stbi_load(argv[3], &width, &height, nullptr, 3);
    if (!data)
    {
        std::cout << "Invalid image path!" << std::endl;
        std::cout << "Terminating program..." << std::endl;
        return 1;
    }
    std::cout << "Success!" << std::endl;

    //checking for correct dimentions
    if (width < 32 || width > 1024 || height < 32 || height > 1024)
    {
        std::cout << "Unsupported image size" << std::endl;
        std::cout << "Terminating program..." << std::endl;
        return 1;
    }
    else
    {
        std::cout << "Loaded image is " << width << " x " << height << " (aspect ratio: " << (double)width / height << ")." << std::endl;
    }

    auto stop0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stop0 - start0);
    std::cout << "Image loading took " << duration0.count() / (double)megaToNormal << " s." << std::endl;

    //CPU version
    std::cout << std::endl << "CPU version of the algorythm:" << std::endl;
    auto start1 = std::chrono::high_resolution_clock::now();

    const char* resultHashCPU = blurHashForPixels(atoi(argv[1]), atoi(argv[2]), width, height, data, width * 3);
    std::string s1 = (std::string)resultHashCPU;

    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    std::cout << "CPU version execution time: " << duration1.count() / (double)megaToNormal << " s" << std::endl;
    std::cout << "Result hash is: " << resultHashCPU << std::endl;

    //GPU version
    std::cout << std::endl << "GPU version of the algorythm:" << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();

    const char* resultHashGPU = blurHashForPixelsCUDA(atoi(argv[1]), atoi(argv[2]), width, height, data, width * 3);
    std::string s2 = (std::string)resultHashGPU;

    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    std::cout << "GPU version execution time: " << duration2.count() / (double)megaToNormal << " s" << std::endl;
    std::cout << "Result hash is: " << resultHashGPU << std::endl;

    //result check
    std::cout << std::endl << "Result check: " << std::endl;
    if (s1 == s2)
    {
        std::cout << "Result hashes are the same :)" << std::endl;
        std::cout << "Success!" << std::endl;
    }
    else
    {
        std::cout << "Result hashes are NOT the same :(" << std::endl;
        std::cout << "Terminating program..." << std::endl;
        return 1;
    }

    stbi_image_free(data);
    std::cout << std::endl;

    return 0;
}