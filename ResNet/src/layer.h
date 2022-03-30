#ifndef LAYER_H

#define LAYER_H

#include <cstdlib>

#include <vector>

#include <memory>

#include <cublas_v2.h>
#include <cuda.h>
#include <sys/time.h>

const static float dt = 1.0E-01f;

const static float threshold = 1.0E-02f;

float offset = 1;

#define Offset 1

#define inputDim 28 * 28

#define c1wDim 5 * 5 * 6
#define c1N 6
#define c1outDim 24 * 24 * 6

#define c2wDim 2 * 2 * 6
#define c2N 6
#define c2outDim 12 * 12 * 6

#define c3wDim 2 * 2 * 6
#define c3N 6
#define c3outDim 6 * 6 * 6

#define fwDim 6 * 6 * 6 * 10
#define fN 10
#define foutDim 10

#define rwDim 4 * 4 * 1
#define rN 1
#define routDim 6 * 6 * 6

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// global mem//
double mallocStart = gettime();
__device__ __managed__ float input_a[inputDim];

__device__ __managed__ float c1_weight[c1wDim];
__device__ __managed__ float c1_bias[c1N];
__device__ __managed__ float c1_a[c1outDim];
__device__ __managed__ float c1_z[c1outDim];

__device__ __managed__ float c2_weight[c2wDim];
__device__ __managed__ float c2_bias[c2N];
__device__ __managed__ float c2_a[c2outDim];
__device__ __managed__ float c2_z[c2outDim];

__device__ __managed__ float c3_weight[c3wDim];
__device__ __managed__ float c3_bias[c3N];
__device__ __managed__ float c3_a[c3outDim];
__device__ __managed__ float c3_z[c3outDim];

__device__ __managed__ float f_weight[fwDim];
__device__ __managed__ float f_bias[fN];
__device__ __managed__ float f_a[foutDim];
__device__ __managed__ float f_z[foutDim];

__device__ __managed__ float r_weight[rwDim];
__device__ __managed__ float r_bias[rN];
__device__ __managed__ float r_a[routDim];
__device__ __managed__ float r_z[routDim];
double mallocEnd = gettime();
#endif

class Layer {
  public:
    int M, N, O;
    float *output;
    float *preact;
    float *bias;
    float *weight;
    float *d_output;
    float *d_preact;
    float *d_weight;
    Layer(int M, int N, int O, char *arg);
    ~Layer();
    void setOutput(float *data);
    void clear();
};

// Utility CUDA kernel functions

__device__ float sigmoid(float v);
float sigmoid_cpu(float v);
__global__ void apply_sigmoid(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5], float *, float);
__global__ void fp_preact_c2(float input[6][24][24], float preact[6][12][12], float weight[6][2][2], float *, float);
__global__ void fp_preact_c3(float input[6][12][12], float preact[6][6][6], float weight[6][2][2], float *, float);
__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6], float *, float);
__global__ void fp_preact_r(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float, float);
__global__ void fp_add_res(float preact1[6][6][6], float preact2[6][6][6]);
////////////////cpu fp kernel////////////////////////////
void fp_preact_c1_cpu(float input[28][28], float preact[6][24][24], float weight[6][5][5], float *);
void fp_preact_c2_cpu(float input[6][24][24], float preact[6][12][12], float weight[6][2][2], float *);
void fp_preact_c3_cpu(float input[6][12][12], float preact[6][6][6], float weight[6][2][2], float *);
void fp_preact_f_cpu(float input[6][6][6], float preact[10], float weight[10][6][6][6], float *);
void fp_preact_r_cpu(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float);
