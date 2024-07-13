#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define WIDTH 1024
#define HEIGHT 1024
#define CHANNELS 3
#define ELEMENT_COUNT (WIDTH*HEIGHT*CHANNELS)

__global__
void colorToGrayscaleConversionKernel(
    unsigned char* Pout, 
    unsigned char* Pin, 
    int width,
    int height,
    int channels
) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(col < width && row < height) {
        int grayOffset = row*width + col;

        int rgbOffset = grayOffset*channels;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void colorToGrayscaleConversion(
    unsigned char* Pout, 
    unsigned char* Pin, 
    int width,
    int height,
    int channels
    ){

    unsigned char *Pout_d, *Pin_d;
    int outSize = sizeof(unsigned char) * width * height;
    int inSize = sizeof(unsigned char) * width * height * channels;

    cudaMalloc((void**) &Pout_d, outSize);
    cudaMalloc((void**) &Pin_d, inSize);

    cudaMemcpy(Pin_d, Pin, inSize, cudaMemcpyHostToDevice);

    dim3 dimGrid(
        ceil((float)width / THREADS_PER_BLOCK_X), 
        ceil((float)height / THREADS_PER_BLOCK_Y), 
        1
    );
    dim3 dimBlock(
        THREADS_PER_BLOCK_X,
        THREADS_PER_BLOCK_Y,
        1
    );

    colorToGrayscaleConversionKernel<<< dimGrid, dimBlock>>>(
        Pout_d,
        Pin_d,
        width, 
        height,
        channels
    );

    // Check for launch errors
    cudaCheckError( cudaGetLastError() );
    // Check for any errors in asynchronous operations
    cudaCheckError( cudaDeviceSynchronize() );

    cudaMemcpy(
        Pout, 
        Pout_d, 
        outSize, 
        cudaMemcpyDeviceToHost
    );

    cudaFree(Pout_d);
    cudaFree(Pin_d);

    return;   
}

int main() {
    const char* input_filename = "data/man.png";
    const char* output_filename = "generated/grayScale.png";

    int width, height, channels;
    unsigned char* Pin = stbi_load(input_filename, &width, &height, &channels, 3);

    if (!Pin) {
        printf("Error loading image %s\n", input_filename);
        return 1;
    }

    printf("Loaded image: %dx%d with %d channels\n", width, height, channels);

    unsigned char* Pout = (unsigned char*)malloc(width * height);

    colorToGrayscaleConversion(
        Pout,
        Pin, 
        width,
        height,
        channels
    );

    // Save the grayscale image
    stbi_write_png(output_filename, width, height, 1, Pout, width);

    printf("Saved grayscale image to %s\n", output_filename);

    // Clean up
    stbi_image_free(Pin);
    free(Pout);

    return 0;
}