#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"

#define TILE_WIDTH 16
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32


__global__ void tiledMatrixMultiplyKernel(
    float *M, 
    float *N,
    float *P,
    int Width
) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int by = blockIdx.y;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float PValue = 0;
    if(Row < Width and Col < Width){
        for (int ph = 0; ph < TILE_WIDTH; ++ph) {

            // Colloborative loading of a tile
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty) * Width + Col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; k++) {
                PValue += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }

        P[Row*Width + Col] = PValue;
    }
}

void tiledMatrixMultiply(
    float *M, 
    float *N, 
    float *P,
    int width
) {
    int nSize = width * width * sizeof(float);
    int mSize = width * width * sizeof(float);
    int pSize = width * width * sizeof(float);

    float *N_d, *M_d, *P_d;

    cudaMalloc((void**) &N_d, nSize);
    cudaMalloc((void**) &M_d, mSize);
    cudaMalloc((void**) &P_d, pSize);

    cudaMemcpy(N_d, N, nSize, cudaMemcpyHostToDevice);
    cudaMemcpy(M_d, M, mSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(
        THREADS_PER_BLOCK_X,
        THREADS_PER_BLOCK_Y,
        1
    );

    dim3 dimGrid(
        ceil(((float) width / TILE_WIDTH) / THREADS_PER_BLOCK_X ),
        ceil(((float) width / TILE_WIDTH) / THREADS_PER_BLOCK_Y ),
        1
    );

    printf("dimBlock: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("dimGrid: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

    tiledMatrixMultiplyKernel<<< dimGrid, dimBlock >>>(N_d, M_d, P_d, width);

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaMemcpy(
        P,
        P_d,
        pSize,
        cudaMemcpyDeviceToHost
    );

    cudaFree(N_d);
    cudaFree(M_d);
    cudaFree(P_d);

    return;
}

int main() {
    int width = 4;
    
    int m_count = width * width;
    int m_size = sizeof(float) * width;

    int n_count = width * width;
    int n_size = sizeof(float) * width;

    int p_count = width * width;
    int p_size = sizeof(float) * width;

    float M[] = {
        1,2,3,4, 
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    };

    float N[] = {
        1,2,3,4, 
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    };

    float P[p_count];

    tiledMatrixMultiply(M, N, P, width);

    printf("P");
    printf("[");
    for (int i = 0; i < width; i++){
        printf("[");
        for (int j = 0; j < width; j++) {
            int index = width*i + j;
            printf("%f,", P[index]);
        }
        printf("],\n");
    }
    printf("]");

    return 0;
}