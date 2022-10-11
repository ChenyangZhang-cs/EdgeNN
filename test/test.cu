#include <stdio.h>
#include <stdlib.h>

#define N 10000000

__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }

__global__ void vector_add(float *out, float *a, float *b) { out[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; }

int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);
    cudaMalloc((void **)&d_a, sizeof(float) * N);
    cudaMalloc((void **)&d_b, sizeof(float) * N);
    cudaMalloc((void **)&d_out, sizeof(float) * N);

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    vector_add<<<N, 1>>>(d_out, d_a, d_b);
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    free(a);
    free(b);
    free(out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    cuda_hello<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}