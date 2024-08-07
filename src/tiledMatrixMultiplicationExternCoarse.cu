#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"

#define THREADS_PER_BLOCK_X 3
#define THREADS_PER_BLOCK_Y 3
#define COARSE_FACTOR 4
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
    int Width,
    int mdsSize, 
    int tile_width
) {
    
    extern __shared__ float Mds_Nds[];
    float *Mds = (float*)Mds_Nds;
    float *Nds = &Mds_Nds[tile_width * tile_width];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int by = blockIdx.y;
    int ty = threadIdx.y;

    int Row = by * tile_width + ty;
    int colStart = bx * tile_width * COARSE_FACTOR + tx;

    float PValue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        PValue[c] = 0.0f;
    }

    int tile_count = (Width + tile_width - 1) / tile_width;
    
    for (int ph = 0; ph < tile_count; ++ph) {
        // Collaborative loading of a tile
        if ((Row < Width) && (ph*tile_width+tx) < Width){
            int mIndex = Row*Width + ph*tile_width + tx;
            float mValue = M[mIndex];
            Mds[ty*tile_width + tx] = mValue;
        } else {
            Mds[ty*tile_width + tx] = 0;
        }

        for (int c = 0; c < COARSE_FACTOR; ++c){    
            int Col = colStart + c*tile_width;

            if ((ph*tile_width+ty) < Width && Col < Width) {
                int nIndex = (ph*tile_width + ty) * Width + Col;
                float nValue = N[nIndex];
                Nds[ty*tile_width + tx] = nValue;
            } else {
                Nds[ty*tile_width + tx] = 0;
            }
            __syncthreads();

            for (int k = 0; k < tile_width; ++k) {
                float mValue = Mds[ty*tile_width + k];
                float nValue = Nds[k*tile_width + tx];
                PValue[c] += mValue * nValue;
            }
            __syncthreads();
        }
        
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int Col = colStart + c*tile_width;
        if(Row < Width && Col < Width){
            P[Row*Width + Col] = PValue[c];
        }
    }
}

size_t appropriate_SM_usage_width(int sharedMemPerBlock){
    // find the closest 
    size_t perArraySize = sharedMemPerBlock / 2;
    float rawElementCount = (float)(perArraySize / sizeof(float));
    float theSqrt = sqrt(rawElementCount);
    size_t width = floor(theSqrt);
    return width;
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

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    // size_t tile_width = appropriate_SM_usage_width(props.sharedMemPerBlock);
    size_t tile_width = 2;
    size_t total_size = 2*tile_width*tile_width * sizeof(float);

    cudaMalloc((void**) &N_d, nSize);
    cudaMalloc((void**) &M_d, mSize);
    cudaMalloc((void**) &P_d, pSize);

    cudaMemcpy(N_d, N, nSize, cudaMemcpyHostToDevice);
    cudaMemcpy(M_d, M, mSize, cudaMemcpyHostToDevice);

    //assert that the tile size is larger than the THREADS_PER_BLOCK_X and 
    // THREADS_PER_BLOCK_Y

    int actual_threads_per_block_x = min(THREADS_PER_BLOCK_X, tile_width);
    int actual_threads_per_block_y = min(THREADS_PER_BLOCK_Y, tile_width);

    dim3 dimBlock(
        actual_threads_per_block_x,
        actual_threads_per_block_y,
        1
    );

    dim3 dimGrid(
        ceil((float) width / actual_threads_per_block_x ),
        ceil((float) width / actual_threads_per_block_y ),
        1
    );


    printf("tile_width %d\n", (int)tile_width);
    printf("dimBlock: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("dimGrid: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

    tiledMatrixMultiplyKernel<<< dimGrid, dimBlock, total_size >>>(
        N_d, M_d, P_d, width, total_size/2, tile_width);


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
    
    int p_count = width * width;

    float M[] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
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