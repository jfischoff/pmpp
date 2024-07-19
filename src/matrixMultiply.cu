#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "errors.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

__global__
void matrixMultiply(float *M, float *N, float *P, int width){ 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width) {
        float pValue = 0;
        for (int k = 0; k < width; k++) {
            pValue += M[row*width + k]*N[k*width + col];
        }

        P[row*width + col] = pValue;
    }
}

void matrixMultiply(float *M, float *N, float *P, int m, int n, int o) {
    float *M_d, *N_d, *P_d;

    int mSize = sizeof(float) * m * n;
    int nSize = sizeof(float) * n * o;
    int pSize = sizeof(float) * m * o;

    cudaMalloc((void**) & M_d, mSize);
    cudaMalloc((void**) & N_d, nSize);
    cudaMalloc((void**) & P_d, pSize);

    cudaMemcpy(M_d, M, mSize, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, nSize, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P, pSize, cudaMemcpyHostToDevice);

    dim3 dimGrid(
        ceil((float) m / THREADS_PER_BLOCK_X),
        ceil((float) o / THREADS_PER_BLOCK_Y),
        1
    );

    dim3 dimBlock(
        THREADS_PER_BLOCK_X,
        THREADS_PER_BLOCK_Y,
        1
    );

    matrixMultiply<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, n);

    // Check for launch errors
    cudaCheckError( cudaGetLastError() );
    // Check for any errors in asynchronous operations
    cudaCheckError( cudaDeviceSynchronize() );

    cudaMemcpy(
        P, 
        P_d,
        pSize,
        cudaMemcpyDeviceToHost
    );

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return;
}

int main() {
    int m = 4;
    int n = 4;
    int o = 4;
    
    int m_count = m * n;
    int m_size = sizeof(float) * m_count;

    int n_count = n * o;
    int n_size = sizeof(float) * n_count;

    int p_count = m * o;
    int p_size = sizeof(float) * p_count;

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

    matrixMultiply(M, N, P, m, n, o);

    printf("P");
    printf("[");
    for (int i = 0; i < m; i++){
        printf("[");
        for (int j = 0; j < o; j++) {
            int index = i*o + j;
            printf("%f,", P[index]);
        }
        printf("],\n");
    }
    printf("]");
}