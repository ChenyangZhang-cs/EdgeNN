#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef MLAYER_H
#define MLAYER_H
#endif

float offset = 1;

#define InDim 32
#define hDim 512
#define OutDim 32
#define Offset 1
#define train_cnt 50
#define test_cnt 100
#define cpurun 1
#define gpurun 1

using namespace std;

const static float dt = 0.5f;
const static float threshold = 1.0E-02f;

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}
// global mem//
double mallocStart = gettime();
__device__ __managed__ float input_a[InDim];

__device__ __managed__ float h_weight[InDim * hDim];
__device__ __managed__ float h_bias[hDim];
__device__ __managed__ float h_a[hDim];
__device__ __managed__ float h_z[hDim];
__device__ __managed__ float h_dweight[InDim * hDim];
__device__ __managed__ float h_da[hDim];
__device__ __managed__ float h_dz[hDim];

__device__ __managed__ float output_weight[OutDim * hDim];
__device__ __managed__ float output_bias[OutDim];
__device__ __managed__ float output_a[OutDim];
__device__ __managed__ float output_z[OutDim];
__device__ __managed__ float output_dweight[OutDim * hDim];
__device__ __managed__ float output_da[OutDim];
__device__ __managed__ float output_dz[OutDim];
double mallocEnd = gettime();

class mLayer {
  public:
    int inDim, outDim;
    float *weight;
    float *bias;
    float *a;
    float *z;
    float *dweight;
    float *da;
    float *dz;

    mLayer(int, int, char *arg = NULL);
    ~mLayer();
    void setOutput0(float *);
    void clear();
    void bp_clear();
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *, float *, float *, const int N = OutDim, float *err = NULL);
__global__ void apply_grad(float *output, float *grad, const int N);
// Forward Propagation
__global__ void fp_z_h(float *input, float *z, float weight[hDim][InDim], float *bias = NULL, float offset = 1);
__global__ void fp_bias_h(float *z, float *bias);
__global__ void fp_z_f(float *input, float *z, float weight[OutDim][hDim], float *bias = NULL, float offset = 1);
__global__ void fp_bias_f(float *z, float *bias);
// corun cpu //
void apply_step_function_cpu(float *input, float *output, const int N);
void makeError_cpu(float *dz, float *a, float *Y, const int N, float *err = NULL);
void apply_grad_cpu(float *output, float *grad, const int N);
void fp_z_h_cpu(float *input, float *z, float weight[hDim][InDim], float *bias, float offset = 0);
void fp_bias_h_cpu(float *z, float *bias);
void fp_z_f_cpu(float *input, float *z, float weight[OutDim][hDim], float *bias, float offset = 0);
void fp_bias_f_cpu(float *z, float *bias);
