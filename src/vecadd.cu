#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000  // Size of vectors
#define THREADS_PER_BLOCK 1024

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void listProps() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("Device %d: %s\n", device, deviceProp.name);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max threads dimension: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max grid size: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("\n");
    }

    return;
}

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n){
    float *A_d, *B_d, *C_d;
    int size = sizeof(float) * n;

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    int gridCount = ceil(n/((float)THREADS_PER_BLOCK));

    vecAddKernel<<<dim3(gridCount, 1, 1), THREADS_PER_BLOCK>>>(A_d, B_d, C_d, size);

    // Check for launch errors
    cudaCheckError( cudaGetLastError() );
    // Check for any errors in asynchronous operations
    cudaCheckError( cudaDeviceSynchronize() );

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    listProps();

    static float A[N], B[N], C[N];

    // Initialize vectors A and B with some values
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Perform vector addition
    vecAdd(A, B, C, N);

    // Print a few results to verify
    printf("A[0] = %.2f, B[0] = %.2f, C[0] = %.2f\n", A[0], B[0], C[0]);
    printf("A[%d] = %.2f, B[%d] = %.2f, C[%d] = %.2f\n", N-1, A[N-1], N-1, B[N-1], N-1, C[N-1]);

    return 0;
}