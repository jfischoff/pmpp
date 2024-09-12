#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PRE_BLOCK_Z 3
#define FILTER_RADIUS 1
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
#define DEBUG_PRINT 0
#if DEBUG_PRINT
#define DPRINTF(...) printf(__VA_ARGS__)
#else
#define DPRINTF(...) do {} while (0)
#endif



__global__ void convolution_2D_basic_kernel(
    float *N, 
    float *P, 
    int r, 
    int width, 
    int height,
    int channels) {

    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int channelIndex = blockIdx.z * blockDim.z + threadIdx.z;

    float Pvalue = 0.0f;

    int filterDiameter = 2*r+1;
    if(outRow < height && outCol < width && channelIndex < channels) {
        for (int fRow = 0; fRow < filterDiameter; fRow++) {
            for (int fCol = 0; fCol < filterDiameter; fCol++) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    
                    float fValue = F[fCol][fRow];
                    float nValue = N[(inRow * width + inCol) * channels + channelIndex];
                    Pvalue += fValue * nValue;
                    
                }
            }
        }
    
        P[(outRow*width + outCol) * channels + channelIndex] = Pvalue;
    }
    
}

__global__ void convolution_2D_tiled_kernel(
    float *N, 
    float *P, 
    int width, 
    int height) {

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    if (row >=0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    int filterDiameter = 2*FILTER_RADIUS+1;

    if (row >=0 && row < height && col >= 0 && col < width) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < filterDiameter; fRow++) {
                for (int fCol = 0; fCol < filterDiameter; fCol++) {
                    Pvalue += F[fCol][fRow] * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width+col] = Pvalue;
        }
    }
}

__global__ void convolution_cached_tiled_2D_const_mem_kernel(
    float *N, 
    float *P, 
    int width, 
    int height) {
    
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y;
    __shared__ float N_s[OUT_TILE_DIM][OUT_TILE_DIM];
    if (row<height && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    int filterDiameter = 2*FILTER_RADIUS+1;

    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for(int fRow = 0; fRow < filterDiameter; fRow++) {
            for(int fCol = 0; fCol < filterDiameter; fCol++) {
                if (threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
                    threadIdx.x - FILTER_RADIUS + fCol < OUT_TILE_DIM &&
                    threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
                    threadIdx.y - FILTER_RADIUS + fRow < OUT_TILE_DIM) {
                    
                    Pvalue += F[fRow][fCol]*N_s[threadIdx.y+fRow][threadIdx.x+fCol];
                } else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height && 
                        col - FILTER_RADIUS + fCol >= 0 && 
                        col - FILTER_RADIUS + fCol < width) {
                        Pvalue += F[fRow][fCol]* N[(row-FILTER_RADIUS+fRow) * width+col - FILTER_RADIUS + fCol];
                    }
                }
            }
        }
        P[row*width+col] = Pvalue;
    }

}

void convolution_2D_basic(
    float *N_h,
    float *F_h,
    float *P_h,
    int r,
    int width,
    int height,
    int channels
) {
    int filterDiameter = r*2+1;
    int fSize = filterDiameter * filterDiameter * sizeof(float); 

    cudaMemcpyToSymbol(F, F_h, fSize);

    float *N, *P;
    int nSize = width * height * channels * sizeof(float);
    int pSize = nSize;
    cudaMalloc((void**) &N, nSize);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaMalloc((void**) &P, pSize);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaMemcpy(N, N_h, nSize, cudaMemcpyHostToDevice);

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    dim3 dimBlock(
        THREADS_PER_BLOCK_X,
        THREADS_PER_BLOCK_Y,
        THREADS_PRE_BLOCK_Z
    );

    dim3 dimGrid(
        ceil((float) width / THREADS_PER_BLOCK_X ),
        ceil((float) height / THREADS_PER_BLOCK_Y ),
        1
    );


    printf("dimBlock: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("dimGrid: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

    convolution_2D_basic_kernel<<< dimGrid, dimBlock >>>(
            N, 
            P, 
            r,
            width, 
            height,
            channels
        );


    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaMemcpy(
        P_h,
        P,
        pSize,
        cudaMemcpyDeviceToHost
    );

    cudaFree(N);
    cudaFree(P);

    return;
}

enum Filter {
    IDENTITY_1,
    BLUR_1,
    BLUR_2
};

enum InputOutput {
    IDENTITY,
    IMAGE
};

int main() {

    Filter filterType = BLUR_1;
    InputOutput inputOutputType = IMAGE;
    int r = -1;
    float* F_h = NULL;
    float F_h_static_1[] = {
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0
    };

    float F_h_static_2[] = {
        0.0751, 0.1238, 0.0751,
        0.1238, 0.2042, 0.1238,
        0.0751, 0.1238, 0.0751
    };

    float F_h_static_3[] = {
        0.025, 0.1, 0.025,
        0.1  , 0.5, 0.1,
        0.25 , 0.1, 0.025
    };

    switch (filterType) {
        case IDENTITY_1:
            r = 1;
            F_h = F_h_static_1;
            break;
        case BLUR_1:
            r = 1;
            F_h = F_h_static_2;
            break;
        case BLUR_2:
            r = 2;
            F_h = F_h_static_3;
            assert(false && "BLUR_2 not implemented");
            break;
    }

    int width, height, channels;
    const char* input_filename = "data/man.png";
    const char* output_filename = "generated/convolution.png";
    unsigned char* Pin = NULL;
    float* N = NULL;
    unsigned char* Pout = NULL;
    float N_static[] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,

        1, 0, 0,
        0, 1, 0,
        0, 0, 1,

        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    

    switch (inputOutputType) {
        case IDENTITY:
            height = 3;
            width = 3;
            channels = 3;
            N = N_static;
            
            break;
        case IMAGE:

            Pin = stbi_load(input_filename, &width, &height, &channels, 3);

            if (!Pin) {
                printf("Error loading image %s\n", input_filename);
                return 1;
            }

            printf("Loaded image: %dx%d with %d channels\n", width, height, channels);

            // convert the unsigned char* to float
            N = (float*)malloc(width * height * channels * sizeof(float));

            // Convert unsigned char to float
            for (int i = 0; i < width * height * channels; i++) {
                N[i] = (float)Pin[i] / 255.0f;
            }
            break;
    }
    
    float* P = (float*)malloc(width * height * channels * sizeof(float));
    int p_count = width * height * channels;

    convolution_2D_basic(
        N, 
        F_h,
        P, 
        r,
        width, 
        height,
        channels
    );

    switch (inputOutputType) {
        case IDENTITY:
            printf("P");
            printf("[");
            for (int i = 0; i < width; i++){
                printf("[");
                for (int j = 0; j < height; j++) {
                    int index = i*width + j;
                    printf("%f,", P[index]);
                }
                printf("],\n");
            }
            printf("]");
            break;
        case IMAGE:
            // Save the grayscale image
            Pout = (unsigned char*)malloc(width * height * channels);
            for (int i = 0; i < width * height * channels; i++) {
                Pout[i] = (unsigned char)(P[i] * 255.0f);
            }
            stbi_write_png(output_filename, width, height, channels, Pout, width * channels);

            printf("Saved grayscale image to %s\n", output_filename);

            // Clean up
            stbi_image_free(Pin);
            free(Pout);
            free(N);
            break;
    }

    free(P);

}