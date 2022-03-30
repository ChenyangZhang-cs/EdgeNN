

#ifndef LAYER_H
#define LAYER_H
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <memory>
#include <sys/time.h>
#include <vector>
#endif
float offset = 1;
#define Offset 1
#define inputDim 227 * 227 * 3
#define c1wDim 11 * 11 * 3 * 2 * 48
#define c1N 96
#define c1outDim 2 * 55 * 55 * 48
#define p1outDim 2 * 31 * 31 * 48
#define c2wDim 5 * 5 * 48 * 2 * 128
#define c2N 256
#define c2outDim 2 * 128 * 27 * 27
#define p2outDim 2 * 15 * 15 * 128
#define c3wDim 3 * 3 * 256 * 384
#define c3N 384
#define c3outDim 2 * 13 * 13 * 192
#define c4wDim 3 * 3 * 192 * 2 * 384
#define c4N 384
#define c4outDim 2 * 13 * 13 * 192
#define c5wDim 3 * 3 * 384 * 2 * 128
#define c5N 256
#define c5outDim 2 * 13 * 13 * 128
#define p3outDim 2 * 6 * 6 * 128
#define f1wDim 6 * 6 * 256 * 2 * 2048
#define f1N 2 * 2048
#define f1outDim 4096
#define f2wDim 4096 * 4096
#define f2N 2 * 2048
#define f2outDim 4096
#define f3wDim 4096 * 1000
#define f3N 1000
#define f3outDim 1000

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
__device__ __managed__ float c1_o[c1outDim];

__device__ __managed__ float p1_a[p1outDim];

__device__ __managed__ float c2_weight[c2wDim];
__device__ __managed__ float c2_bias[c2N];
__device__ __managed__ float c2_a[c2outDim];
__device__ __managed__ float c2_z[c2outDim];
__device__ __managed__ float c2_o[c2outDim];

__device__ __managed__ float p2_a[p2outDim];

__device__ __managed__ float c3_weight[c3wDim];
__device__ __managed__ float c3_bias[c3N];
__device__ __managed__ float c3_a[c3outDim];
__device__ __managed__ float c3_z[c3outDim];

__device__ __managed__ float c4_weight[c4wDim];
__device__ __managed__ float c4_bias[c4N];
__device__ __managed__ float c4_a[c4outDim];
__device__ __managed__ float c4_z[c4outDim];

__device__ __managed__ float c5_weight[c5wDim];
__device__ __managed__ float c5_bias[c5N];
__device__ __managed__ float c5_a[c5outDim];
__device__ __managed__ float c5_z[c5outDim];

__device__ __managed__ float p3_a[p3outDim];

__device__ __managed__ float f1_weight[f1wDim];
__device__ __managed__ float f1_bias[f1N];
__device__ __managed__ float f1_a[f1outDim];
__device__ __managed__ float f1_z[f1outDim];

__device__ __managed__ float f2_weight[f2wDim];
__device__ __managed__ float f2_bias[f2N];
__device__ __managed__ float f2_a[f2outDim];
__device__ __managed__ float f2_z[f2outDim];

__device__ __managed__ float f3_weight[f3wDim];
__device__ __managed__ float f3_bias[f3N];
__device__ __managed__ float f3_a[f3outDim];
__device__ __managed__ float f3_z[f3outDim];
double mallocEnd = gettime();

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

typedef struct pool_data {
    unsigned int x;
    unsigned int y;
} pool_data; // for pooling layer thread x and y
class Layer {
  public:
    bool isLRN;
    long int M, N, O;

    float *output;
    float *preact;
    float *act_result; // for normalization

    float *bias;
    float *weight;

    float *L_output;

    float *d_output;
    float *d_preact;
    float *d_act_result; // for normalization
    float *d_weight;

    Layer(long int M, long int N, long int O, char *arg);

    ~Layer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
    void Output_Layer(float *data);
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
float step_function_cpu(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
void apply_step_function_cpu(float *input, float *output, const int N);
__device__ float normalization(float *input, float u, int idx, const int O, const int N);
float normalization_cpu(float *input, float u, int idx, const int O, const int N);
__global__ void normalization_function(float *input, float *output, const int O, const int N);
void normalization_function_cpu(float *input, float *output, const int O, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[227][227][3], float preact[96][55][55], float weight[96][11][11][3], float *);
__global__ void fp_bias_c1(float preact[96][55][55], float bias[96]);
__global__ void fp_preact_p1(float input[96][55][55], float preact[96][31][31]);
//__global__ void fp_bias_p1(float preact[6][12][12], float bias[1]);
__global__ void fp_preact_c2(float input[96][31][31], float preact[128][27][27], float weight[128][96][5][5], float *);
__global__ void fp_bias_c2(float preact[128][27][27], float bias[128]);
__global__ void fp_preact_p2(float input[256][27][27], float preact[256][15][15]);
__global__ void fp_preact_c3(float input[256][15][15], float preact[384][13][13], float weight[384][256][3][3],
                             float *);
__global__ void fp_bias_c3(float preact[384][13][13], float bias[384]);
__global__ void fp_preact_c4(float input[384][13][13], float preact[384][13][13], float weight[384][384][3][3],
                             float *);
__global__ void fp_bias_c4(float preact[384][13][13], float bias[384]);
__global__ void fp_preact_c5(float input[384][13][13], float preact[256][13][13], float weight[256][384][3][3],
                             float *);
__global__ void fp_bias_c5(float preact[256][13][13], float bias[256]);
__global__ void fp_preact_p3(float input[256][13][13], float preact[256][6][6]);
__global__ void fp_preact_f1(float input[256][6][6], float preact[4096], float weight[4096][256][6][6], float *);
__global__ void fp_bias_f1(float preact[4096], float bias[4096]);
__global__ void fp_preact_f2(float input[4096], float preact[4096], float weight[4096][4096], float *);
__global__ void fp_bias_f2(float preact[4096], float bias[4096]);
__global__ void fp_preact_f3(float input[4096], float preact[1000], float weight[1000][4096], float *);
__global__ void fp_bias_f3(float preact[1000], float bias[1000]);
