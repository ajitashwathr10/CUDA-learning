#include <stdio.h>

__global__ void HelloWorld() {
    printf("Hello World from Thread %d in Block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello World (from CPU)!\n");
    
    HelloWorld<<<2, 4>>>();
    //Launching kernel with 2 blocks (with 4 threads each)
    cudaDeviceSynchronize();
    //Waiting for GPU to finish
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        return -1;
    }
    printf("Goodbye World (from CPU)!\n"); 
    return 0;
}
