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

    cudaMemcpy(Pout_d, Pout, outSize, cudaMemcpyHostToDevice);
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
        cudaMemcpyHostToDevice
    );

    cudaFree(Pout_d);

    return;   
}

int main() {

    static unsigned char Pin[ELEMENT_COUNT], Pout[WIDTH*HEIGHT];

    unsigned char pixelValue = 0;
    for(int i = 0; i < WIDTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            for(int k = 0; k < CHANNELS; k++){
                pixelValue = (pixelValue + 1) % 256;

                int  index = WIDTH * i + HEIGHT * j + k;
                Pin[index] = pixelValue;
            }
        }
    }

    colorToGrayscaleConversion(
        Pout,
        Pin, 
        WIDTH,
        HEIGHT,
        CHANNELS
    );



}