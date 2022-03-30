#include "layer.h"

#include "omp.h"

// Constructor

Layer::Layer(int M, int N, int O, char *arg) {
    this->M = M;
    this->N = N;
    this->O = O;
    output = NULL;
    preact = NULL;
    bias = NULL;
    weight = NULL;
    d_output = NULL;
    d_preact = NULL;
    d_weight = NULL;

    if (strcmp(arg, "input") == 0)
        output = input_a;
    else if (strcmp(arg, "c1") == 0) {
        output = c1_a;
        preact = c1_z;
        bias = c1_bias;
        weight = c1_weight;

    } else if (strcmp(arg, "c2") == 0) {
        output = c2_a;
        preact = c2_z;
        bias = c2_bias;
        weight = c2_weight;
    } else if (strcmp(arg, "c3") == 0) {
        output = c3_a;
        preact = c3_z;
        bias = c3_bias;
        weight = c3_weight;
    } else if (strcmp(arg, "f") == 0) {
        output = f_a;
        preact = f_z;
        bias = f_bias;
        weight = f_weight;
    } else if (strcmp(arg, "r") == 0) {
        output = r_a;
        preact = r_z;
        bias = r_bias;
        weight = r_weight;
    }

    if (M)
        for (int i = 0; i < N; ++i) {
            bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
            for (int j = 0; j < M; ++j) {
                weight[i * M + j] = 0.5f - float(rand()) / float(RAND_MAX);
            }
        }
}

Layer::~Layer() {
    cudaFree(output);
    cudaFree(preact);
    cudaFree(bias);
    cudaFree(weight);
    if (d_output)
        cudaFree(d_output);
    if (d_preact)
        cudaFree(d_preact);
    if (d_weight)
        cudaFree(d_weight);
}

// Send data one row from dataset to the GPU

void Layer::setOutput(float *data) { memcpy(output, data, sizeof(float) * O); }

// Reset GPU memory between iterations

void Layer::clear() {
    memset(output, 0x00, sizeof(float) * O);
    if (preact)
        memset(preact, 0x00, sizeof(float) * O);
}

__device__ float sigmoid(float v) { return 1 / (1 + exp(-v)); }

float sigmoid_cpu(float v) { return 1 / (1 + exp(-v)); }

__global__ void apply_sigmoid(float *input, float *output, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] = sigmoid(input[idx]);
    }
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
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

////////////////////////gpu kernel///////////////////////////////////////////////
// conv
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5], float *bias,
                             float offset = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= int(6 * 24 * 24 * offset))
        return;
    const int i3 = ((idx) % 6);
    const int i4 = ((idx /= 6) % 24);
    const int i5 = (idx /= 24);
    float temp = bias[i3];
    for (int i1 = 0; i1 < 5; i1++)
        for (int i2 = 0; i2 < 5; i2++) {
            temp += weight[i3][i1][i2] * input[i4 + i1][i5 + i2];
        }

    temp = sigmoid(temp);
    preact[i3][i4][i5] = temp;
}

// pool with weight
__global__ void fp_preact_c2(float input[6][24][24], float preact[6][12][12], float weight[6][2][2], float *bias,
                             float offset = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= int(6 * 12 * 12 * offset))
        return;
    const int i3 = (idx % 6);
    const int i4 = ((idx /= 6) % 12);
    const int i5 = (idx /= 12);
    float temp = bias[i3];
    for (int i1 = 0; i1 < 2; i1++)
        for (int i2 = 0; i2 < 2; i2++) {
            temp += weight[i3][i1][i2] * input[i3][i4 * 2 + i1][i5 * 2 + i2];
        }

    temp = sigmoid(temp);
    preact[i3][i4][i5] = temp;
}

// pool with weight
__global__ void fp_preact_c3(float input[6][12][12], float preact[6][6][6], float weight[6][2][2], float *bias,
                             float offset = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= int(6 * 6 * 6 * offset))
        return;
    const int i3 = (idx % 6);
    const int i4 = ((idx /= 6) % 6);
    const int i5 = (idx /= 6);
    float temp = bias[i3];
    for (int i1 = 0; i1 < 2; i1++)
        for (int i2 = 0; i2 < 2; i2++) {
            temp += weight[i3][i1][i2] * input[i3][i4 * 2 + i1][i5 * 2 + i2];
        }
    preact[i3][i4][i5] = temp;
}

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6], float *bias,
                            float offset = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= int(10 * offset))
        return;
    const int i1 = idx;
    float temp = bias[i1];
    for (int i2 = 0; i2 < 6; i2++)
        for (int i3 = 0; i3 < 6; i3++)
            for (int i4 = 0; i4 < 6; i4++) {
                temp += weight[i1][i2][i3][i4] * input[i2][i3][i4];
            }
    temp = sigmoid(temp);
    preact[i1] = temp;
}

__global__ void fp_add_res(float preact1[6][6][6], float preact2[6][6][6]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 6 * 6 * 6)
        return;
    const int i1 = (idx % 6);
    const int i2 = ((idx /= 6) % 6);
    const int i3 = idx /= 6;
    preact1[i1][i2][i3] += preact2[i1][i2][i3];
}

// another type of pool
__global__ void fp_preact_r(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float bias,
                            float offset = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= int(6 * 6 * 6 * offset))
        return;
    const int i3 = (idx % 6);
    const int i4 = ((idx /= 6) % 6);
    const int i5 = idx /= 6;
    float temp = bias;
    for (int i1 = 0; i1 < 4; i1++)
        for (int i2 = 0; i2 < 4; i2++) {
            temp += weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2];
        }
    preact[i3][i4][i5] = temp;
}

//////////////////////////cpu kernel////////////////////////////////////////////////////
void fp_preact_c1_cpu(float input[28][28], float preact[6][24][24], float weight[6][5][5], float *bias) {
    const int N = 6 * 24 * 24;
#pragma omp parallel for
    for (int idx = int(N * offset); idx < N; idx++) {
        int tempi = idx;
        int i3 = ((tempi) % 6);
        int i4 = ((tempi /= 6) % 24);
        int i5 = (tempi /= 24);
        float temp = bias[i3];
        for (int i1 = 0; i1 < 5; i1++)
            for (int i2 = 0; i2 < 5; i2++) {
                temp += weight[i3][i1][i2] * input[i4 + i1][i5 + i2];
            }

        temp = sigmoid_cpu(temp);
        preact[i3][i4][i5] = temp;
    }
}

// pool with weight
void fp_preact_c2_cpu(float input[6][24][24], float preact[6][12][12], float weight[6][2][2], float *bias) {
    const int N = 6 * 12 * 12;
#pragma omp parallel for
    for (int idx = int(N * offset); idx < N; idx++) {
        int tempi = idx;
        int i3 = (tempi % 6);
        int i4 = ((tempi /= 6) % 12);
        int i5 = (tempi /= 12);
        float temp = bias[i3];
        for (int i1 = 0; i1 < 2; i1++)
            for (int i2 = 0; i2 < 2; i2++) {
                temp += weight[i3][i1][i2] * input[i3][i4 * 2 + i1][i5 * 2 + i2];
            }
        temp = sigmoid_cpu(temp);
        preact[i3][i4][i5] = temp;
    }
}

// pool with weight
void fp_preact_c3_cpu(float input[6][12][12], float preact[6][6][6], float weight[6][2][2], float *bias) {
    const int N = 6 * 6 * 6;
#pragma omp parallel for
    for (int idx = int(N * offset); idx < N; idx++) {
        int tempi = idx;
        int i3 = (tempi % 6);
        int i4 = ((tempi /= 6) % 6);
        int i5 = (tempi /= 6);
        float temp = bias[i3];
        for (int i1 = 0; i1 < 2; i1++)
            for (int i2 = 0; i2 < 2; i2++) {
                temp += weight[i3][i1][i2] * input[i3][i4 * 2 + i1][i5 * 2 + i2];
            }
        preact[i3][i4][i5] = temp;
    }
}

void fp_preact_f_cpu(float input[6][6][6], float preact[10], float weight[10][6][6][6], float *bias) {
    const int N = 10;
#pragma omp parallel for
    for (int i1 = int(N * offset); i1 < N; i1++) {
        float temp = bias[i1];
        for (int i2 = 0; i2 < 6; i2++)
            for (int i3 = 0; i3 < 6; i3++)
                for (int i4 = 0; i4 < 6; i4++) {
                    temp += weight[i1][i2][i3][i4] * input[i2][i3][i4];
                }
        temp = sigmoid_cpu(temp);
        preact[i1] = temp;
    }
}

// another type of pool
void fp_preact_r_cpu(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float bias) {
    const int N = 6 * 6 * 6;
#pragma omp parallel for
    for (int idx = int(N * offset); idx < N; idx++) {
        int tempi = idx;
        int i3 = (tempi % 6);
        int i4 = ((tempi /= 6) % 6);
        int i5 = (tempi /= 6);
        float temp = bias;
        for (int i1 = 0; i1 < 4; i1++)
            for (int i2 = 0; i2 < 4; i2++) {
                temp += weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2];
            }
        preact[i3][i4][i5] = temp;
    }
}