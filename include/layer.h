#ifndef LAYER_H
#define LAYER_H
#include <bits/stdc++.h>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <memory>
#include <sys/time.h>
#include <vector>
#endif

float offset = 1;
#define Offset 1
#define AinputDim 227 * 227 * 3
#define Ac1wDim 11 * 11 * 3 * 2 * 48
#define Ac1N 96
#define Ac1outDim 2 * 55 * 55 * 48
#define Ap1outDim 2 * 31 * 31 * 48
#define Ac2wDim 5 * 5 * 48 * 2 * 128
#define Ac2N 256
#define Ac2outDim 2 * 128 * 27 * 27
#define Ap2outDim 2 * 15 * 15 * 128
#define Ac3wDim 3 * 3 * 256 * 384
#define Ac3N 384
#define Ac3outDim 2 * 13 * 13 * 192
#define Ac4wDim 3 * 3 * 192 * 2 * 384
#define Ac4N 384
#define Ac4outDim 2 * 13 * 13 * 192
#define Ac5wDim 3 * 3 * 384 * 2 * 128
#define Ac5N 256
#define Ac5outDim 2 * 13 * 13 * 128
#define Ap3outDim 2 * 6 * 6 * 128
#define Af1wDim 6 * 6 * 256 * 2 * 2048
#define Af1N 2 * 2048
#define Af1outDim 4096
#define Af2wDim 4096 * 4096
#define Af2N 2 * 2048
#define Af2outDim 4096
#define Af3wDim 4096 * 1000
#define Af3N 1000
#define Af3outDim 1000

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// global mem//
double mallocStart = gettime();
__device__ __managed__ float Ainput_a[AinputDim];

__device__ __managed__ float Ac1_weight[Ac1wDim];
__device__ __managed__ float Ac1_bias[Ac1N];
__device__ __managed__ float Ac1_a[Ac1outDim];
__device__ __managed__ float Ac1_z[Ac1outDim];
__device__ __managed__ float Ac1_o[Ac1outDim];

__device__ __managed__ float Ap1_a[Ap1outDim];

__device__ __managed__ float Ac2_weight[Ac2wDim];
__device__ __managed__ float Ac2_bias[Ac2N];
__device__ __managed__ float Ac2_a[Ac2outDim];
__device__ __managed__ float Ac2_z[Ac2outDim];
__device__ __managed__ float Ac2_o[Ac2outDim];

__device__ __managed__ float Ap2_a[Ap2outDim];

__device__ __managed__ float Ac3_weight[Ac3wDim];
__device__ __managed__ float Ac3_bias[Ac3N];
__device__ __managed__ float Ac3_a[Ac3outDim];
__device__ __managed__ float Ac3_z[Ac3outDim];

__device__ __managed__ float Ac4_weight[Ac4wDim];
__device__ __managed__ float Ac4_bias[Ac4N];
__device__ __managed__ float Ac4_a[Ac4outDim];
__device__ __managed__ float Ac4_z[Ac4outDim];

__device__ __managed__ float Ac5_weight[Ac5wDim];
__device__ __managed__ float Ac5_bias[Ac5N];
__device__ __managed__ float Ac5_a[Ac5outDim];
__device__ __managed__ float Ac5_z[Ac5outDim];

__device__ __managed__ float Ap3_a[Ap3outDim];

__device__ __managed__ float Af1_weight[Af1wDim];
__device__ __managed__ float Af1_bias[Af1N];
__device__ __managed__ float Af1_a[Af1outDim];
__device__ __managed__ float Af1_z[Af1outDim];

__device__ __managed__ float Af2_weight[Af2wDim];
__device__ __managed__ float Af2_bias[Af2N];
__device__ __managed__ float Af2_a[Af2outDim];
__device__ __managed__ float Af2_z[Af2outDim];

__device__ __managed__ float Af3_weight[Af3wDim];
__device__ __managed__ float Af3_bias[Af3N];
__device__ __managed__ float Af3_a[Af3outDim];
__device__ __managed__ float Af3_z[Af3outDim];
double mallocEnd = gettime();

const static float Adt = 1.0E-01f;
const static float Athreshold = 1.0E-02f;

typedef struct pool_data {
    unsigned int x;
    unsigned int y;
} pool_data; // for pooling layer thread x and y
class ALayer {
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

    ALayer(long int M, long int N, long int O, char *arg);

    ~ALayer();

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


// FCNN

#define InDim 32
#define hDim 512
#define OutDim 32
#define train_cnt 50
#define test_cnt 100
#define cpurun 1
#define gpurun 1

using namespace std;

const static float Fdt = 0.5f;
const static float Fthreshold = 1.0E-02f;

// global mem//
double FmallocStart = gettime();
__device__ __managed__ float Finput_a[InDim];

__device__ __managed__ float Fh_weight[InDim * hDim];
__device__ __managed__ float Fh_bias[hDim];
__device__ __managed__ float Fh_a[hDim];
__device__ __managed__ float Fh_z[hDim];
__device__ __managed__ float Fh_dweight[InDim * hDim];
__device__ __managed__ float Fh_da[hDim];
__device__ __managed__ float Fh_dz[hDim];

__device__ __managed__ float Foutput_weight[OutDim * hDim];
__device__ __managed__ float Foutput_bias[OutDim];
__device__ __managed__ float Foutput_a[OutDim];
__device__ __managed__ float Foutput_z[OutDim];
__device__ __managed__ float Foutput_dweight[OutDim * hDim];
__device__ __managed__ float Foutput_da[OutDim];
__device__ __managed__ float Foutput_dz[OutDim];
double FmallocEnd = gettime();

class FLayer {
  public:
    int inDim, outDim;
    float *weight;
    float *bias;
    float *a;
    float *z;
    float *dweight;
    float *da;
    float *dz;

    FLayer(int, int, char *arg = NULL);
    ~FLayer();
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


// LeNet
#define DEVICE 0
#define LinputDim 28 * 28
#define Lc1wDim 5 * 5
#define Lc1N 6
#define Lc1outDim 24 * 24 * 6
#define Ls1wDim 2 * 2
#define Ls1N 1
#define Ls1outDim 12 * 12 * 6
#define Lc2wDim 5 * 5 * 6
#define Lc2N 16
#define Lc2outDim 8 * 8 * 16
#define Ls2wDim 2 * 2
#define Ls2N 1
#define Ls2outDim 4 * 4 * 16
#define Lc3wDim 4 * 4 * 16
#define Lc3N 120
#define Lc3outDim 1 * 1 * 120
#define Lf1wDim 120
#define Lf1N 84
#define Lf1outDim 84
#define Lf2wDim 84
#define Lf2N 10
#define Lf2outDim 10

const static float Ldt = 1.0E-01f;
const static float Lthreshold = 1.0E-02f;

// global mem//
double LmallocStart = gettime();
__device__ __managed__ float Lc3_weight[Lc3wDim];
__device__ __managed__ float Lc3_bias[Lc3N];
__device__ __managed__ float Lc3_a[Lc3outDim];
__device__ __managed__ float Lc3_z[Lc3outDim];
__device__ __managed__ float Lc3_dweight[Lc3wDim];
__device__ __managed__ float Lc3_da[Lc3outDim];
__device__ __managed__ float Lc3_dz[Lc3outDim];

__device__ __managed__ float Linput_a[LinputDim];

__device__ __managed__ float Lc1_weight[Lc1wDim];
__device__ __managed__ float Lc1_bias[Lc1N];
__device__ __managed__ float Lc1_a[Lc1outDim];
__device__ __managed__ float Lc1_z[Lc1outDim];
__device__ __managed__ float Lc1_dweight[Lc1wDim];
__device__ __managed__ float Lc1_da[Lc1outDim];
__device__ __managed__ float Lc1_dz[Lc1outDim];

__device__ __managed__ float Ls1_weight[Ls1wDim];
__device__ __managed__ float Ls1_bias[Ls1N];
__device__ __managed__ float Ls1_a[Ls1outDim];
__device__ __managed__ float Ls1_z[Ls1outDim];
__device__ __managed__ float Ls1_dweight[Ls1wDim];
__device__ __managed__ float Ls1_da[Ls1outDim];
__device__ __managed__ float Ls1_dz[Ls1outDim];

__device__ __managed__ float Lc2_weight[Lc2wDim];
__device__ __managed__ float Lc2_bias[Lc2N];
__device__ __managed__ float Lc2_a[Lc2outDim];
__device__ __managed__ float Lc2_z[Lc2outDim];
__device__ __managed__ float Lc2_dweight[Lc2wDim];
__device__ __managed__ float Lc2_da[Lc2outDim];
__device__ __managed__ float Lc2_dz[Lc2outDim];

__device__ __managed__ float Ls2_weight[Ls2wDim];
__device__ __managed__ float Ls2_bias[Ls2N];
__device__ __managed__ float Ls2_a[Ls2outDim];
__device__ __managed__ float Ls2_z[Ls2outDim];
__device__ __managed__ float Ls2_dweight[Ls2wDim];
__device__ __managed__ float Ls2_da[Ls2outDim];
__device__ __managed__ float Ls2_dz[Ls2outDim];

__device__ __managed__ float Lf1_weight[Lf1wDim];
__device__ __managed__ float Lf1_bias[Lf1N];
__device__ __managed__ float Lf1_a[Lf1outDim];
__device__ __managed__ float Lf1_z[Lf1outDim];
__device__ __managed__ float Lf1_dweight[Lf1wDim];
__device__ __managed__ float Lf1_da[Lf1outDim];
__device__ __managed__ float Lf1_dz[Lf1outDim];

__device__ __managed__ float Lf2_weight[Lf2wDim];
__device__ __managed__ float Lf2_bias[Lf2N];
__device__ __managed__ float Lf2_a[Lf2outDim];
__device__ __managed__ float Lf2_z[Lf2outDim];
__device__ __managed__ float Lf2_dweight[Lf2wDim];
__device__ __managed__ float Lf2_da[Lf2outDim];
__device__ __managed__ float Lf2_dz[Lf2outDim];
double LmallocEnd = gettime();

class LLayer {
  public:
    int M, N, O;

    float *output; // a
    float *preact; // z

    float *bias;
    float *weight;

    float *d_output; // da
    float *d_preact; // dz
    float *d_weight; // dw

    LLayer(int M, int N, int O, int, double &);

    ~LLayer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);


// ResNet
const static float Rdt = 1.0E-01f;

const static float Rthreshold = 1.0E-02f;

#define RinputDim 28 * 28

#define Rc1wDim 5 * 5 * 6
#define Rc1N 6
#define Rc1outDim 24 * 24 * 6

#define Rc2wDim 2 * 2 * 6
#define Rc2N 6
#define Rc2outDim 12 * 12 * 6

#define Rc3wDim 2 * 2 * 6
#define Rc3N 6
#define Rc3outDim 6 * 6 * 6

#define RfwDim 6 * 6 * 6 * 10
#define RfN 10
#define RfoutDim 10

#define RrwDim 4 * 4 * 1
#define RrN 1
#define RroutDim 6 * 6 * 6

// global mem//
double RmallocStart = gettime();
__device__ __managed__ float Rinput_a[RinputDim];

__device__ __managed__ float Rc1_weight[Rc1wDim];
__device__ __managed__ float Rc1_bias[Rc1N];
__device__ __managed__ float Rc1_a[Rc1outDim];
__device__ __managed__ float Rc1_z[Rc1outDim];

__device__ __managed__ float Rc2_weight[Rc2wDim];
__device__ __managed__ float Rc2_bias[Rc2N];
__device__ __managed__ float Rc2_a[Rc2outDim];
__device__ __managed__ float Rc2_z[Rc2outDim];

__device__ __managed__ float Rc3_weight[Rc3wDim];
__device__ __managed__ float Rc3_bias[Rc3N];
__device__ __managed__ float Rc3_a[Rc3outDim];
__device__ __managed__ float Rc3_z[Rc3outDim];

__device__ __managed__ float Rf_weight[RfwDim];
__device__ __managed__ float Rf_bias[RfN];
__device__ __managed__ float Rf_a[RfoutDim];
__device__ __managed__ float Rf_z[RfoutDim];

__device__ __managed__ float Rr_weight[RrwDim];
__device__ __managed__ float Rr_bias[RrN];
__device__ __managed__ float Rr_a[RroutDim];
__device__ __managed__ float Rr_z[RroutDim];
double RmallocEnd = gettime();

class RLayer {
  public:
    int M, N, O;
    float *output;
    float *preact;
    float *bias;
    float *weight;
    float *d_output;
    float *d_preact;
    float *d_weight;
    RLayer(int M, int N, int O, char *arg);
    ~RLayer();
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
