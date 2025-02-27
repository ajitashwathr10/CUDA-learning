#include <stdio.h>
#include <stdlib.h>

__global__ void VectorAdd(const float *A, const float *B, float *C, int num) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num) C[i] = A[i] + B[i];
    // Condition to check out of bounds
}

int main() {
    int num = 50000;
    size_t size = num * sizeof(float);
    //Hosting CPU vectors
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    //Input vectors
    for (int i = 0; i < num; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Using 256 threads per block
    int threads = 256;
    int blocks = (num + threads - 1) / threads;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks, threads);
    VectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, num);
    // Check for errors (if there)
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "VectorAdd launch failed: %s\n", cudaGetErrorString(status));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        return -1;
    }
    
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(status));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        return -1;
    }
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Verifying result\n");
    for (int i = 0; i < 5; i++) {
        printf("h_A[%d] = %f, h_B[%d] = %f, h_C[%d] = %f\n", i, h_A[i], i, h_B[i], i, h_C[i]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
