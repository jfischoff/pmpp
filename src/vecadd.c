__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIndex.x + blockDim.x * blockInd.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* b, float* c, int n){
    float *A_d, *B_d, *C_d;
    int size = sizeof(float) * n;

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d);

    cudaMemcpy(C, C_d, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}