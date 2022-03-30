#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

float offset = 1;
//#define offset 1
#define DEVICE 0
#define gpurun 1
#define cpurun 1
#define inputDim 28 * 28
#define c1wDim 5 * 5
#define c1N 6
#define c1outDim 24 * 24 * 6
#define s1wDim 2 * 2
#define s1N 1
#define s1outDim 12 * 12 * 6
#define c2wDim 5 * 5 * 6
#define c2N 16
#define c2outDim 8 * 8 * 16
#define s2wDim 2 * 2
#define s2N 1
#define s2outDim 4 * 4 * 16
#define c3wDim 4 * 4 * 16
#define c3N 120
#define c3outDim 1 * 1 * 120
#define f1wDim 120
#define f1N 84
#define f1outDim 84
#define f2wDim 84
#define f2N 10
#define f2outDim 10

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;
double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// global mem//
double mallocStart = gettime();
__device__ __managed__ float c3_weight[c3wDim];
__device__ __managed__ float c3_bias[c3N];
__device__ __managed__ float c3_a[c3outDim];
__device__ __managed__ float c3_z[c3outDim];
__device__ __managed__ float c3_dweight[c3wDim];
__device__ __managed__ float c3_da[c3outDim];
__device__ __managed__ float c3_dz[c3outDim];

__device__ __managed__ float input_a[inputDim];

__device__ __managed__ float c1_weight[c1wDim];
__device__ __managed__ float c1_bias[c1N];
__device__ __managed__ float c1_a[c1outDim];
__device__ __managed__ float c1_z[c1outDim];
__device__ __managed__ float c1_dweight[c1wDim];
__device__ __managed__ float c1_da[c1outDim];
__device__ __managed__ float c1_dz[c1outDim];

__device__ __managed__ float s1_weight[s1wDim];
__device__ __managed__ float s1_bias[s1N];
__device__ __managed__ float s1_a[s1outDim];
__device__ __managed__ float s1_z[s1outDim];
__device__ __managed__ float s1_dweight[s1wDim];
__device__ __managed__ float s1_da[s1outDim];
__device__ __managed__ float s1_dz[s1outDim];

__device__ __managed__ float c2_weight[c2wDim];
__device__ __managed__ float c2_bias[c2N];
__device__ __managed__ float c2_a[c2outDim];
__device__ __managed__ float c2_z[c2outDim];
__device__ __managed__ float c2_dweight[c2wDim];
__device__ __managed__ float c2_da[c2outDim];
__device__ __managed__ float c2_dz[c2outDim];

__device__ __managed__ float s2_weight[s2wDim];
__device__ __managed__ float s2_bias[s2N];
__device__ __managed__ float s2_a[s2outDim];
__device__ __managed__ float s2_z[s2outDim];
__device__ __managed__ float s2_dweight[s2wDim];
__device__ __managed__ float s2_da[s2outDim];
__device__ __managed__ float s2_dz[s2outDim];

__device__ __managed__ float f1_weight[f1wDim];
__device__ __managed__ float f1_bias[f1N];
__device__ __managed__ float f1_a[f1outDim];
__device__ __managed__ float f1_z[f1outDim];
__device__ __managed__ float f1_dweight[f1wDim];
__device__ __managed__ float f1_da[f1outDim];
__device__ __managed__ float f1_dz[f1outDim];

__device__ __managed__ float f2_weight[f2wDim];
__device__ __managed__ float f2_bias[f2N];
__device__ __managed__ float f2_a[f2outDim];
__device__ __managed__ float f2_z[f2outDim];
__device__ __managed__ float f2_dweight[f2wDim];
__device__ __managed__ float f2_da[f2outDim];
__device__ __managed__ float f2_dz[f2outDim];
double mallocEnd = gettime();

class Layer {
  public:
    int M, N, O;

    float *output; // a
    float *preact; // z

    float *bias;
    float *weight;

    float *d_output; // da
    float *d_preact; // dz
    float *d_weight; // dw

    Layer(int M, int N, int O, int, double &);

    ~Layer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);
