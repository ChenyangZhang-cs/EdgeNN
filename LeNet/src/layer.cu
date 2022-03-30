#include "layer.h"

using namespace std;

Layer::Layer(int M, int N, int O, int arg, double &randTime) {
    this->M = M;
    this->N = N;
    this->O = O;

    if (arg == 0)
        output = input_a;
    else if (arg == 1) {
        output = c1_a;
        preact = c1_z;
        bias = c1_bias;
        weight = c1_weight;
        d_output = c1_da;
        d_preact = c1_dz;
        d_weight = c1_dweight;
    } else if (arg == 2) {
        output = s1_a;
        preact = s1_z;
        bias = s1_bias;
        weight = s1_weight;
        d_output = s1_da;
        d_preact = s1_dz;
        d_weight = s1_dweight;
    } else if (arg == 3) {
        output = c2_a;
        preact = c2_z;
        bias = c2_bias;
        weight = c2_weight;
        d_output = c2_da;
        d_preact = c2_dz;
        d_weight = c2_dweight;
    } else if (arg == 4) {
        output = s2_a;
        preact = s2_z;
        bias = s2_bias;
        weight = s2_weight;
        d_output = s2_da;
        d_preact = s2_dz;
        d_weight = s2_dweight;
    } else if (arg == 5) {
        output = c3_a;
        preact = c3_z;
        bias = c3_bias;
        weight = c3_weight;
        d_output = c3_da;
        d_preact = c3_dz;
        d_weight = c3_dweight;
    } else if (arg == 6) {
        output = f1_a;
        preact = f1_z;
        bias = f1_bias;
        weight = f1_weight;
        d_output = f1_da;
        d_preact = f1_dz;
        d_weight = f1_dweight;
    } else if (arg == 7) {
        output = f2_a;
        preact = f2_z;
        bias = f2_bias;
        weight = f2_weight;
        d_output = f2_da;
        d_preact = f2_dz;
        d_weight = f2_dweight;
    }
    double randSt = gettime();
    if (M)
        for (int i = 0; i < N; ++i) {
            bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
            for (int j = 0; j < M; ++j) {
                weight[i * M + j] = 0.5f - float(rand()) / float(RAND_MAX);
            }
        }
    double randEnd = gettime();
    randTime += randEnd - randSt;
}

Layer::~Layer() {
    cudaFree(output);
    cudaFree(preact);
    cudaFree(bias);
    cudaFree(weight);
    cudaFree(d_output);
    cudaFree(d_preact);
    cudaFree(d_weight);
}

void Layer::setOutput(float *data) { memcpy(output, data, sizeof(float) * O); }

void Layer::clear() {
    memset(output, 0x00, sizeof(float) * O);
    if (M)
        memset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear() {
    memset(d_output, 0x00, sizeof(float) * O);
    memset(d_preact, 0x00, sizeof(float) * O);
    memset(d_weight, 0x00, sizeof(float) * M * N);
}

__device__ float step_function(float v) // Sigmoid function::Activation Function
{
    return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] = step_function(input[idx]);
    }
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x; // find specific index/thread in GPU
    const int size = blockDim.x * gridDim.x;               // the size of all index/thread in GPU

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
    }
}

__global__ void apply_grad(float *output, float *grad, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] += dt * grad[idx];
    }
}

void get_round(int &round) {
    round = 200;
    if (fabs(offset - 0.9) < 0.01)
        round = 200 * 0.4;
    if (fabs(offset - 0.8) < 0.01)
        round = 200 * 0.32;
    if (fabs(offset - 0.7) < 0.01)
        round = 200 * 0.31;
    if (fabs(offset - 0.6) < 0.01)
        round = 200 * 0.45;
    if (fabs(offset - 0.5) < 0.01)
        round = 200 * 0.7;
}