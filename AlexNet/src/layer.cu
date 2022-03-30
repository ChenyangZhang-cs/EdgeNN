#ifndef __LAYER1__
#define __LAYER1__
#include "layer.h"
#include <cstdio>
#include <omp.h>
#endif
//#include "mnist.h"

// Constructor
Layer::Layer(long int M, long int N, long int O, char *arg) {
    isLRN = 0;
    this->M = M;
    this->N = N;
    this->O = O;

    output = NULL;
    preact = NULL;
    act_result = NULL; // for normalization
    bias = NULL;
    weight = NULL;
    L_output = NULL; // output result need to be copied from GPU to CPU to print out
    d_output = NULL;
    d_preact = NULL;
    d_act_result = NULL; // for normalization
    d_weight = NULL;

    if (strcmp(arg, "input") == 0)
        output = input_a;
    else if (strcmp(arg, "c1") == 0) {
        isLRN = 1;
        // output=c1_o;
        cudaMalloc(&output, sizeof(float) * 2 * 55 * 55 * 48);
        act_result = c1_a;
        preact = c1_z;
        bias = c1_bias;
        weight = c1_weight;

    } else if (strcmp(arg, "p1") == 0) {
        output = p1_a;
    } else if (strcmp(arg, "c2") == 0) {
        isLRN = 1;
        output = c2_o;
        act_result = c2_a;
        preact = c2_z;
        bias = c2_bias;
        weight = c2_weight;
    } else if (strcmp(arg, "p2") == 0) {
        output = p2_a;
    } else if (strcmp(arg, "c3") == 0) {
        output = c3_a;
        preact = c3_z;
        bias = c3_bias;
        weight = c3_weight;
    } else if (strcmp(arg, "c4") == 0) {
        output = c4_a;
        preact = c4_z;
        bias = c4_bias;
        weight = c4_weight;
    } else if (strcmp(arg, "c5") == 0) {
        output = c5_a;
        preact = c5_z;
        bias = c5_bias;
        weight = c5_weight;
    } else if (strcmp(arg, "p3") == 0) {
        output = p3_a;
    } else if (strcmp(arg, "f1") == 0) {
        output = f1_a;
        preact = f1_z;
        bias = f1_bias;
        weight = f1_weight;
    } else if (strcmp(arg, "f2") == 0) {
        output = f2_a;
        preact = f2_z;
        bias = f2_bias;
        weight = f2_weight;
    } else if (strcmp(arg, "f3") == 0) {
        output = f3_a;
        preact = f3_z;
        bias = f3_bias;
        weight = f3_weight;
    }
    if (M && arg[0] != 'p')
        for (int i = 0; i < N; ++i) {
            bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

            for (int j = 0; j < M; ++j) {
                weight[i * M + j] = 0.5f - float(rand()) / float(RAND_MAX);
            }
        }
}

// Destructor
Layer::~Layer() {
    cudaFree(output);
    cudaFree(preact);
    if (isLRN)
        cudaFree(act_result); // for normalization

    cudaFree(bias);

    cudaFree(weight);
    if (d_output)
        cudaFree(d_output);
    if (d_preact)
        cudaFree(d_preact);
    if (d_act_result)
        cudaFree(d_act_result); // for normalization
    if (d_weight)
        cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data) { memcpy(output, data, sizeof(float) * O); }

// Reset GPU memory between iterations
void Layer::clear() {
    if (output)
        memset(output, 0x00, sizeof(float) * O);
    if (preact)
        memset(preact, 0x00, sizeof(float) * O);
    if (isLRN)
        memset(act_result, 0x00, sizeof(float) * O); // for normalization
}

void Layer::bp_clear() {
    memset(d_output, 0x00, sizeof(float) * O);
    memset(d_preact, 0x00, sizeof(float) * O);
    if (isLRN)
        memset(d_act_result, 0x00, sizeof(float) * O); // for normalization
    memset(d_weight, 0x00, sizeof(float) * M * N);
}

void Layer::Output_Layer(float *data) { cudaMemcpy(L_output, data, sizeof(float) * O, cudaMemcpyDeviceToHost); }

__device__ float step_function(float v) // Sigmoid function::Activation Function
{
    return max(0.f, v);
}
float step_function_cpu(float v) // Sigmoid function::Activation Function
{
    return max(0.f, v);
}

__device__ float normalization(float *input, float u, int idx, const int O, const int N) {

    int i = ((idx / (O / N)) % N);
    int j, k;
    float tmp_sum = 0.f;
    if ((i - (5 / 2)) > 0) {
        j = i - (5 / 2);
    } else {
        j = 0;
    }
    if ((i + (5 / 2)) > (N - 1)) {
        k = (N - 1);
    } else {
        k = (i + (5 / 2));
    }
    for (int tmp_i = j; tmp_i < k; tmp_i++) {
        tmp_sum += pow(input[(idx - (i - 1) * (O / N)) + ((j - 1) * (O / N))], 2);
    }
    return u / pow((2 + (0.0001 * tmp_sum)), 0.75);
} // normalization

float normalization_cpu(float *input, float u, int idx, const int O, const int N) {

    int i = ((idx / (O / N)) % N);
    int j, k;
    float tmp_sum = 0.f;
    if ((i - (5 / 2)) > 0) {
        j = i - (5 / 2);
    } else {
        j = 0;
    }
    if ((i + (5 / 2)) > (N - 1)) {
        k = (N - 1);
    } else {
        k = (i + (5 / 2));
    }
    for (int tmp_i = j; tmp_i < k; tmp_i++) {
        tmp_sum += pow(input[(idx - (i - 1) * (O / N)) + ((j - 1) * (O / N))], 2);
    }
    return u / pow((2 + (0.0001 * tmp_sum)), 0.75);
} // normalization

__global__ void apply_step_function(float *input, float *output, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    // int endN=N*offset;

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] = step_function(input[idx]);
    }
}
void apply_step_function_cpu(float *input, float *output, const int N) {
    for (int idx = N * offset; idx < N; ++idx) {
        output[idx] = step_function_cpu(input[idx]);
    }
}

__global__ void normalization_function(float *input, float *output, const int O, const int N) {

    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    for (int idx = O * pos / size; idx < O * (pos + 1) / size; ++idx) {
        output[idx] = normalization(input, input[idx], idx, O, N);
    }

} // normalization

void normalization_function_cpu(float *input, float *output, const int O, const int N) {
    for (int idx = 0; idx < O; ++idx) {
        output[idx] = normalization_cpu(input, input[idx], idx, O, N);
    }

} // normalization

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

__global__ void fp_preact_c1(float input[227][227][3], float preact[96][55][55], float weight[96][11][11][3],
                             float bias[96]) {
    int i3 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i3];
    for (int i2 = 0; i2 < 11; i2++)
        for (int i1 = 0; i1 < 11; i1++)
            for (int i6 = 0; i6 < 3; i6++) {
                temp += weight[i3][i1][i2][i6] * input[i4 * 4 + i1][i5 * 4 + i2][i6];
            }
    preact[i3][i4][i5] = temp;
}

__global__ void fp_preact_p1(float input[96][55][55], float preact[96][31][31]) {
    pool_data pos;
    pos.x = threadIdx.x;
    pos.y = threadIdx.y;
    float tmp_preact = input[blockIdx.x][2 * pos.x][2 * pos.y];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; j++) {
            if (tmp_preact < input[blockIdx.x][2 * pos.x + i][2 * pos.y + j]) {
                tmp_preact = input[blockIdx.x][2 * pos.x + i][2 * pos.y + j];
            }
        }
    }
    preact[blockIdx.x][pos.x + 2][pos.y + 2] = tmp_preact;
}

__global__ void fp_preact_c2(float input[96][31][31], float preact[256][27][27], float weight[256][96][5][5],
                             float bias[256]) {

    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 96; i3++)
        for (int i2 = 0; i2 < 5; i2++)
            for (int i1 = 0; i1 < 5; i1++)
                temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
    preact[i6][i4][i5] = temp;
}

__global__ void fp_preact_p2(float input[256][27][27], float preact[256][15][15]) {
    pool_data pos;
    pos.x = threadIdx.x;
    pos.y = threadIdx.y;
    float tmp_preact = input[blockIdx.x][2 * pos.x][2 * pos.y];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; j++) {
            if (tmp_preact < input[blockIdx.x][2 * pos.x + i][2 * pos.y + j]) {
                tmp_preact = input[blockIdx.x][2 * pos.x + i][2 * pos.y + j];
            }
        }
    }
    preact[blockIdx.x][pos.x + 2][pos.y + 2] = tmp_preact;
}

__global__ void fp_preact_c3(float input[256][15][15], float preact[384][13][13], float weight[384][256][3][3],
                             float bias[384]) {
    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 256; i3++)
        for (int i2 = 0; i2 < 3; i2++)
            for (int i1 = 0; i1 < 3; i1++) {
                temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
            }
    preact[i6][i4][i5] = temp;
}

__global__ void fp_preact_c4(float input[384][13][13], float preact[384][13][13], float weight[384][384][3][3],
                             float bias[384]) {
    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 384; i3++)
        for (int i2 = 0; i2 < 3; i2++)
            for (int i1 = 0; i1 < 3; i1++) {
                temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
            }
    preact[i6][i4 + 1][i5 + 1] = temp;
}

__global__ void fp_preact_c5(float input[384][13][13], float preact[256][13][13], float weight[256][384][3][3],
                             float bias[256]) {
    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 384; i3++)
        for (int i2 = 0; i2 < 3; i2++)
            for (int i1 = 0; i1 < 3; i1++) {
                temp += weight[i6][i3][i1][i2] *
                        input[i3][i4 + i1][i5 + i2]; // after convolution, dim is 11*11, so plus 1 to fill the center
            }
    preact[i6][i4 + 1][i5 + 1] = temp;
}

__global__ void fp_preact_p3(float input[256][13][13], float preact[256][6][6]) {
    pool_data pos;
    pos.x = threadIdx.x;
    pos.y = threadIdx.y;
    float tmp_preact = input[blockIdx.x][2 * pos.x][2 * pos.y];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; j++) {
            if (tmp_preact < input[blockIdx.x][2 * pos.x + i][2 * pos.y + j]) {
                tmp_preact = input[blockIdx.x][2 * pos.x + i][2 * pos.y + j];
            }
        }
    }
    preact[blockIdx.x][pos.x][pos.y] = tmp_preact;
}

// full connect 1 6*6*256 to 4096*1*1
__global__ void fp_preact_f1(float input[256][6][6], float preact[4096], float weight[4096][256][6][6],
                             float bias[4096]) {
    int i6 = blockIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 256; i3++)
        for (int i2 = 0; i2 < 6; i2++)
            for (int i1 = 0; i1 < 6; i1++) {
                temp += weight[i6][i3][i1][i2] * input[i3][i1][i2];
            }
    preact[i6] = temp;
}

// full connect 2 4096 to 4096
__global__ void fp_preact_f2(float input[4096], float preact[4096], float weight[4096][4096], float bias[4096]) {
    int i1 = blockIdx.x;
    float temp = bias[i1];
    for (int i2 = 0; i2 < 4096; i2++) {
        temp += weight[i1][i2] * input[i2];
    }
    preact[i1] = temp;
}

// full connect 3 4096 to 1000
__global__ void fp_preact_f3(float input[4096], float preact[1000], float weight[1000][4096], float bias[1000]) {
    int i1 = blockIdx.x;
    float temp = bias[i1];
    for (int i2 = 0; i2 < 4096; i2++) {
        temp += weight[i1][i2] * input[i2];
    }
    preact[i1] = temp;
}

/////////////////////////////////////CPU Forward Propagation//////////////////////////////////////////////

// conv1 227*227*3 to 55*55*96
void fp_preact_c1_cpu(float input[227][227][3], float preact[96][55][55], float weight[96][11][11][3], float bias[96]) {

#pragma omp parallel for
    for (int i3 = int(96 * offset); i3 < 96; i3++)
        for (int i4 = 0; i4 < 55; i4++)
            for (int i5 = 0; i5 < 55; i5++) {
                float temp = bias[i3];
                for (int i1 = 0; i1 < 11; i1++)
                    for (int i2 = 0; i2 < 11; i2++)
                        for (int i6 = 0; i6 < 3; i6++) {
                            temp += weight[i3][i1][i2][i6] * input[i4 * 4 + i1][i5 * 4 + i2][i6];
                        }
                preact[i3][i4][i5] = temp;
            }
}

void fp_bias_c1_cpu(float preact[96][55][55], float bias[96]) {
    const int N = 96 * 55 * 55, startN = int(96 * offset) * 55 * 55;
    for (int n = startN; n < N; ++n) {
        int idx = n;
        const int i2 = ((idx /= 1) % 55);
        const int i3 = ((idx /= 55) % 55);
        const int i1 = ((idx /= 55) % 96);
        preact[i1][i2][i3] += bias[i1];
    }
}

// pooling 1 55*55*96 to 31*31*96
void fp_preact_p1_cpu(float input[96][55][55], float preact[96][31][31]) {
    for (int blockIdx = 0; blockIdx < 96; blockIdx++)
        for (int posy = 0; posy < 27; posy++)
            for (int posx = 0; posx < 27; posx++) {
                float tmp_preact = input[blockIdx][2 * posx][2 * posy];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; j++) {
                        if (tmp_preact < input[blockIdx][2 * posx + i][2 * posy + j]) {
                            tmp_preact = input[blockIdx][2 * posx + i][2 * posy + j];
                        }
                    }
                }
                preact[blockIdx][posx + 2][posy + 2] = tmp_preact;
            }
}

// conv2 31*31*96 to 27*27*128
void fp_preact_c2_cpu(float input[96][31][31], float preact[256][27][27], float weight[256][96][5][5],
                      float bias[256]) {
#pragma omp parallel for
    for (int i6 = int(256 * offset); i6 < 256; i6++)
        for (int i4 = 0; i4 < 27; i4++)
            for (int i5 = 0; i5 < 27; i5++) {
                float temp = bias[i6];
                for (int i3 = 0; i3 < 96; i3++)
                    for (int i1 = 0; i1 < 5; i1++)
                        for (int i2 = 0; i2 < 5; i2++) {
                            temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
                        }
                preact[i6][i4][i5] = temp;
            }
}

void fp_bias_c2_cpu(float preact[256][27][27], float bias[256]) {
    const int N = 256 * 27 * 27, startN = int(256 * offset) * 27 * 27;
    for (int n = startN; n < N; ++n) {
        int idx = n;
        const int i2 = ((idx /= 1) % 27);
        const int i3 = ((idx /= 27) % 27);
        const int i1 = ((idx /= 27) % 256);
        preact[i1][i2][i3] += bias[i1];
    }
}

// pooling 2 27*27*128 to 15*15*128
void fp_preact_p2_cpu(float input[256][27][27], float preact[256][15][15]) {
    for (int blockIdx = 0; blockIdx < 256; blockIdx++)
        for (int posy = 0; posy < 13; posy++)
            for (int posx = 0; posx < 13; posx++) {
                float tmp_preact = input[blockIdx][2 * posx][2 * posy];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; j++) {
                        if (tmp_preact < input[blockIdx][2 * posx + i][2 * posy + j]) {
                            tmp_preact = input[blockIdx][2 * posx + i][2 * posy + j];
                        }
                    }
                }
                preact[blockIdx][posx + 2][posy + 2] = tmp_preact;
            }
}

// conv3 256*15*15 to 13*13*384
void fp_preact_c3_cpu(float input[256][15][15], float preact[384][13][13], float weight[384][256][3][3],
                      float bias[384]) {
#pragma omp parallel for
    for (int i6 = int(384 * offset); i6 < 384; i6++)
        for (int i4 = 0; i4 < 13; i4++)
            for (int i5 = 0; i5 < 13; i5++) {
                float temp = bias[i6];
                for (int i3 = 0; i3 < 256; i3++)
                    for (int i1 = 0; i1 < 3; i1++)
                        for (int i2 = 0; i2 < 3; i2++) {
                            temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
                        }
                preact[i6][i4][i5] = temp;
            }
}

void fp_bias_c3_cpu(float preact[384][13][13], float bias[384]) {
    const int N = 384 * 13 * 13, startN = int(384 * offset) * 13 * 13;
    for (int n = startN; n < N; ++n) {
        int idx = n;
        const int i2 = ((idx /= 1) % 13);
        const int i3 = ((idx /= 13) % 13);
        const int i1 = ((idx /= 13) % 384);
        preact[i1][i2][i3] += bias[i1];
    }
}

void fp_preact_c4_cpu(float input[384][13][13], float preact[384][13][13], float weight[384][384][3][3],
                      float bias[384]) {
#pragma omp parallel for
    for (int i6 = int(384 * offset); i6 < 384; i6++)
        for (int i4 = 0; i4 < 11; i4++)
            for (int i5 = 0; i5 < 11; i5++) {
                float temp = bias[i6];
                for (int i3 = 0; i3 < 384; i3++)
                    for (int i1 = 0; i1 < 3; i1++)
                        for (int i2 = 0; i2 < 3; i2++) {
                            temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
                        }
                preact[i6][i4 + 1][i5 + 1] = temp;
            }
}

void fp_bias_c4_cpu(float preact[384][13][13], float bias[384]) {
    const int N = 384 * 13 * 13, startN = int(384 * offset) * 13 * 13;
    for (int n = startN; n < N; ++n) {
        int idx = n;
        const int i2 = ((idx /= 1) % 13);
        const int i3 = ((idx /= 13) % 13);
        const int i1 = ((idx /= 13) % 384);
        preact[i1][i2][i3] += bias[i1];
    }
}

void fp_preact_c5_cpu(float input[384][13][13], float preact[256][13][13], float weight[256][384][3][3],
                      float bias[256]) {
#pragma omp parallel for
    for (int i6 = int(256 * offset); i6 < 256; i6++)
        for (int i4 = 0; i4 < 11; i4++)
            for (int i5 = 0; i5 < 11; i5++) {
                float temp = bias[i6];
                for (int i3 = 0; i3 < 384; i3++)
                    for (int i1 = 0; i1 < 3; i1++)
                        for (int i2 = 0; i2 < 3; i2++) {
                            temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
                        }
                preact[i6][i4 + 1][i5 + 1] = temp;
            }
}

void fp_bias_c5_cpu(float preact[256][13][13], float bias[256]) {
    const int N = 256 * 13 * 13, startN = int(256 * offset) * 13 * 13;
    for (int n = startN; n < N; ++n) {
        int idx = n;
        const int i2 = ((idx /= 1) % 13);
        const int i3 = ((idx /= 13) % 13);
        const int i1 = ((idx /= 13) % 256);
        preact[i1][i2][i3] += bias[i1];
    }
}

void fp_preact_p3_cpu(float input[256][13][13], float preact[256][6][6]) {
    for (int blockIdx = 0; blockIdx < 256; blockIdx++)
        for (int posy = 0; posy < 6; posy++)
            for (int posx = 0; posx < 6; posx++) {
                float tmp_preact = input[blockIdx][2 * posx][2 * posy];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; j++) {
                        if (tmp_preact < input[blockIdx][2 * posx + i][2 * posy + j]) {
                            tmp_preact = input[blockIdx][2 * posx + i][2 * posy + j];
                        }
                    }
                }
                preact[blockIdx][posx + 2][posy + 2] = tmp_preact;
            }
}

// full connect 1 6*6*256 to 4096*1*1
void fp_preact_f1_cpu(float input[256][6][6], float preact[4096], float weight[4096][256][6][6], float bias[4096]) {
#pragma omp parallel for
    for (int i6 = int(4096 * offset); i6 < 4096; i6++) {
        float temp = bias[i6];
        for (int i3 = 0; i3 < 256; i3++)
            for (int i2 = 0; i2 < 6; i2++)
                for (int i1 = 0; i1 < 6; i1++) {
                    temp += weight[i6][i3][i1][i2] * input[i3][i1][i2];
                }
        preact[i6] = temp;
    }
}
void fp_bias_f1_cpu(float preact[4096], float bias[4096]) {
    const int N = 4096, startN = 4096 * offset;
    for (int idx = startN; idx < N; ++idx) {
        preact[idx] += bias[idx];
    }
}

// full connect 2 4096 to 4096
void fp_preact_f2_cpu(float input[4096], float preact[4096], float weight[4096][4096], float bias[4096]) {
#pragma omp parallel for
    for (int i1 = int(4096 * offset); i1 < 4096; i1++) {
        float temp = bias[i1];
        for (int i2 = 0; i2 < 4096; i2++) {
            temp += weight[i1][i2] * input[i2];
        }
        preact[i1] = temp;
    }
}

void fp_bias_f2_cpu(float preact[4096], float bias[4096]) {
    const int N = 4096, startN = 4096 * offset;
    for (int idx = startN; idx < N; ++idx) {
        preact[idx] += bias[idx];
    }
}

// full connect 3 4096 to 1000
void fp_preact_f3_cpu(float input[4096], float preact[1000], float weight[1000][4096], float bias[1000]) {
#pragma omp parallel for
    for (int i1 = int(1000 * offset); i1 < 1000; i1++) {
        float temp = bias[i1];
        for (int i2 = 0; i2 < 4096; i2++) {
            temp += weight[i1][i2] * input[i2];
        }
        preact[i1] = temp;
    }
}

void fp_bias_f3_cpu(float preact[1000], float bias[1000]) {
    const int N = 1000, startN = 1000 * offset;
    for (int idx = startN; idx < N; ++idx) {
        preact[idx] += bias[idx];
    }
}
