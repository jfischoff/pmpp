#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"

#define TILE_WIDTH 4 
#define THREADS_PER_BLOCK_X 3
#define THREADS_PER_BLOCK_Y 3
#define min(a,b) ((a) < (b) ? (a) : (b))

void slow_matrix_multiplication(float *m, float *n, float *p, int width){
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float pValue = 0;
            for (int k = 0; k < width; ++k) {
                int nIndex = width*row + k;
                int mIndex = width*k + col;

                pValue += m[mIndex]*n[nIndex];
            }
            int pIndex = width*row + col;
            p[pIndex] = pValue;
        }
    }
}

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

    int tile_count = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int ph = 0; ph < tile_count; ++ph) {
        // Collaborative loading of a tile
        if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width){
            int mIndex = Row*Width + ph*TILE_WIDTH + tx;
            float mValue = M[mIndex];
            Mds[ty][tx] = mValue;
        } else {
            Mds[ty][tx] = 0;
        }

        if ((ph*TILE_WIDTH+ty) < Width && Col < Width) {
            int nIndex = (ph*TILE_WIDTH + ty) * Width + Col;
            float nValue = N[nIndex];
            Nds[ty][tx] = nValue;
        } else {
            Nds[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            float mValue = Mds[ty][k];
            float nValue = Nds[k][tx];
            PValue += mValue * nValue;
        }
        __syncthreads();
        
    }

    if(Row < Width && Col < Width){
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
        ceil((float) width / THREADS_PER_BLOCK_X ),
        ceil((float) width / THREADS_PER_BLOCK_Y ),
        1
    );

    //assert that the tile size is larger than the THREADS_PER_BLOCK_X and 
    // THREADS_PER_BLOCK_Y
    assert(THREADS_PER_BLOCK_X <= TILE_WIDTH);
    assert(THREADS_PER_BLOCK_X <= TILE_WIDTH);

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
    int width = 3;
    
    int m_count = width * width;
    int m_size = sizeof(float) * width;

    int n_count = width * width;
    int n_size = sizeof(float) * width;

    int p_count = width * width;
    int p_size = sizeof(float) * width;

    float M[] = {
        1,0,0,
        0,1,0,
        0,0,1
    };

    float N[] = {
        1,2,3,
        4,5,6,
        7,8,9
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
    printf("]\n");

    float slowP[p_count];
    slow_matrix_multiplication(M, N, slowP, width);

    bool mismatch = false;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            int pIndex = i*width + j;
            float cudaValue = P[pIndex];
            float slowValue = slowP[pIndex];
            if (cudaValue != slowValue) {
                printf("Index %d %d was %f for CUDA and %f for slow\n", i, j, cudaValue, slowValue);
                mismatch = true;
            }
        }
    }

    return mismatch ? 1 : 0;
}