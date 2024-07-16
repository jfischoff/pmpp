#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"


#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define THREADS_PER_BLOCK_Z 4


__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h, int c, int blurSize) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int channel = blockDim.z * blockIdx.z + threadIdx.z;

    if (col < w && row < h && channel < c) {
        int pixVal = 0;
        int pixelCount = 0;

        for (int blurRow = -blurSize; blurRow < blurSize + 1; blurRow++) {
            int curRow = row + blurRow;
            int rowOffset = curRow*w;
            
            for (int blurCol = -blurSize; blurCol < blurSize + 1; blurCol++) {
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[(rowOffset + curCol) * c + channel];
                    ++pixelCount;
                }
            }
        }

        out[(row*w + col) * c + channel] = (unsigned char)(pixVal / pixelCount);
    }
}

void blur(unsigned char* in, unsigned char *out, int w, int h, int c, int blurSize) {
    unsigned char *in_d, *out_d;

    assert(c < 4);

    int inSize = sizeof(unsigned char) * w * h * c;
    int outSize = inSize;

    cudaMalloc((void**) &in_d, inSize);
    cudaMalloc((void**) &out_d, outSize);

    cudaMemcpy(in_d, in, inSize, cudaMemcpyHostToDevice);

    dim3 dimGrid(
        ceil((float)w / THREADS_PER_BLOCK_X),
        ceil((float)h / THREADS_PER_BLOCK_Y),
        4
    );
    printf("dimGrid: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    
    dim3 dimBlock(
        THREADS_PER_BLOCK_X,
        THREADS_PER_BLOCK_Y,
        1
    );
    printf("dimBlock: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);


    blurKernel<<< dimGrid, dimBlock>>>(in_d, out_d, w, h, c, blurSize);

    // Check for launch errors
    cudaCheckError( cudaGetLastError() );
    // Check for any errors in asynchronous operations
    cudaCheckError( cudaDeviceSynchronize() );

    cudaMemcpy(
        out,
        out_d,
        outSize, 
        cudaMemcpyDeviceToHost
    );

    cudaFree(in_d);
    cudaFree(out_d);

    return;
}

int main() {
    const char* input_filename = "data/man.png";
    const char* output_filename = "generated/blurred.png";
    int blurSize = 100;

    int width, height, channels;
    unsigned char* Pin = stbi_load(input_filename, &width, &height, &channels, 3);

    if (!Pin) {
        printf("Error loading image %s\n", input_filename);
        return 1;
    }

    printf("Loaded image: %dx%d with %d channels\n", width, height, channels);

    unsigned char* Pout = (unsigned char*)malloc(width * height * channels);
    if (blurSize > 1) {
        blur(
            Pin,
            Pout, 
            width, 
            height,
            channels,
            blurSize
        );
    } else {
        memcpy(Pout, Pin, width*height*channels);
    }

    stbi_write_png(output_filename, width, height, channels, Pout, width * channels);

    printf("Saved blurred image to %s\n", output_filename);
    stbi_image_free(Pin);
    free(Pout);

    return 0;
}