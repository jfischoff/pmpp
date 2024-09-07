#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"

#define THREADS_PER_BLOCK_X 3
#define THREADS_PER_BLOCK_Y 3
#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];



__global__ void convolution_2D_basic_kernel(
    float *N, 
    float *P, 
    int r, 
    int width, 
    int height) {

    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;

    int filterDiameter = 2*r+1;

    for (int fRow = 0; fRow < filterDiameter; fRow++) {
        for (int fCol = 0; fCol < filterDiameter; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow > 0 && inRow < height && inCol > 0 && inCol < width) {
                Pvalue += F[fCol][fRow] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow*width + outCol] = Pvalue;
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
    int height
) {
    //int nSize = 

    int filterDiameter = r*2+1;
    int fSize = filterDiameter * filterDiameter * sizeof(float); 

    cudaMemcpyToSymbol(F, F_h, fSize);

    float *N, *P;
    int nSize = width * height * sizeof(float);
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
        1
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
            height
        );


    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaMemcpy(
        P,
        P_h,
        pSize,
        cudaMemcpyDeviceToHost
    );

    cudaFree(N_h);
    cudaFree(P_h);

    return;
}

int main() {

    int height = 3;
    int width = 3;
    int p_count = width * height;

    float N[] = {
        0,0,0,
        0,1,0,
        0,0,0
    };

    float F_h[] = {
        0.025, 0.1, 0.025,
        0.1  , 0.5, 0.1,
        0.25 , 0.1, 0.025
    };

    float P[p_count]; 

    int r = 2;



    convolution_2D_basic(
        N, 
        F_h,
        P, 
        r,
        width, 
        height
    );

    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            int pIndex = i*width + j;
            float cudaValue = P[pIndex];
            printf("i %d, j %d, value %f", i, j, cudaValue);
        }
    }
}