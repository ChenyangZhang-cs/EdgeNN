#include "../include/layer.h"
#include <cstdio>
#include <omp.h>

using namespace std;

// AlexNet

// Constructor
ALayer::ALayer(long int M, long int N, long int O, char *arg) {
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
        output = Ainput_a;
    else if (strcmp(arg, "c1") == 0) {
        isLRN = 1;
        // output=c1_o;
        cudaMalloc(&output, sizeof(float) * 2 * 55 * 55 * 48);
        act_result = Ac1_a;
        preact = Ac1_z;
        bias = Ac1_bias;
        weight = Ac1_weight;

    } else if (strcmp(arg, "p1") == 0) {
        output = Ap1_a;
    } else if (strcmp(arg, "c2") == 0) {
        isLRN = 1;
        output = Ac2_o;
        act_result = Ac2_a;
        preact = Ac2_z;
        bias = Ac2_bias;
        weight = Ac2_weight;
    } else if (strcmp(arg, "p2") == 0) {
        output = Ap2_a;
    } else if (strcmp(arg, "c3") == 0) {
        output = Ac3_a;
        preact = Ac3_z;
        bias = Ac3_bias;
        weight = Ac3_weight;
    } else if (strcmp(arg, "c4") == 0) {
        output = Ac4_a;
        preact = Ac4_z;
        bias = Ac4_bias;
        weight = Ac4_weight;
    } else if (strcmp(arg, "c5") == 0) {
        output = Ac5_a;
        preact = Ac5_z;
        bias = Ac5_bias;
        weight = Ac5_weight;
    } else if (strcmp(arg, "p3") == 0) {
        output = Ap3_a;
    } else if (strcmp(arg, "f1") == 0) {
        output = Af1_a;
        preact = Af1_z;
        bias = Af1_bias;
        weight = Af1_weight;
    } else if (strcmp(arg, "f2") == 0) {
        output = Af2_a;
        preact = Af2_z;
        bias = Af2_bias;
        weight = Af2_weight;
    } else if (strcmp(arg, "f3") == 0) {
        output = Af3_a;
        preact = Af3_z;
        bias = Af3_bias;
        weight = Af3_weight;
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
ALayer::~ALayer() {
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
void ALayer::setOutput(float *data) { memcpy(output, data, sizeof(float) * O); }

// Reset GPU memory between iterations
void ALayer::clear() {
    if (output)
        memset(output, 0x00, sizeof(float) * O);
    if (preact)
        memset(preact, 0x00, sizeof(float) * O);
    if (isLRN)
        memset(act_result, 0x00, sizeof(float) * O); // for normalization
}

void ALayer::bp_clear() {
    memset(d_output, 0x00, sizeof(float) * O);
    memset(d_preact, 0x00, sizeof(float) * O);
    if (isLRN)
        memset(d_act_result, 0x00, sizeof(float) * O); // for normalization
    memset(d_weight, 0x00, sizeof(float) * M * N);
}

void ALayer::Output_Layer(float *data) { cudaMemcpy(L_output, data, sizeof(float) * O, cudaMemcpyDeviceToHost); }

__device__ float Astep_function(float v) // Sigmoid function::Activation Function
{
    return max(0.f, v);
}
float Astep_function_cpu(float v) // Sigmoid function::Activation Function
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

__global__ void Aapply_step_function(float *input, float *output, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    // int endN=N*offset;

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] = Astep_function(input[idx]);
    }
}
void Aapply_step_function_cpu(float *input, float *output, const int N) {
    for (int idx = N * offset; idx < N; ++idx) {
        output[idx] = Astep_function_cpu(input[idx]);
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

__global__ void AmakeError(float *err, float *output, unsigned int Y, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x; // find specific index/thread in GPU
    const int size = blockDim.x * gridDim.x;               // the size of all index/thread in GPU

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
    }
}

__global__ void Aapply_grad(float *output, float *grad, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] += Adt * grad[idx];
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


// FCNN

FLayer::FLayer(int in, int out, char *arg) {
    this->inDim = in;
    this->outDim = out;

    if (arg != NULL && strcmp(arg, "input") == 0)
        a = Finput_a;

    if (arg != NULL && (strcmp(arg, "h") == 0 || strcmp(arg, "htest") == 0)) {
        weight = Fh_weight;
        bias = Fh_bias;
        a = Fh_a;
        z = Fh_z;
        dweight = Fh_dweight;
        da = Fh_da;
        dz = Fh_dz;
    }

    if (arg != NULL && (strcmp(arg, "output") == 0 || strcmp(arg, "outputtest") == 0)) {
        weight = Foutput_weight;
        bias = Foutput_bias;
        a = Foutput_a;
        z = Foutput_z;
        dweight = Foutput_dweight;
        da = Foutput_da;
        dz = Foutput_dz;
    }

    if (arg != NULL && (strcmp(arg, "test") == 0 || strcmp(arg, "htest") == 0 || strcmp(arg, "outputtest") == 0)) {
        weight[0] = 0.4;
        weight[2] = 0.5;
        weight[1] = 0.45;
        weight[3] = 0.55;
        bias[0] = 0.35;
        bias[1] = 0.6;
    }

    else if (arg == NULL || strcmp(arg, "input") != 0)
        for (int i = 0; i < out; i++) {
            bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

            for (int j = 0; j < in; j++) {
                weight[i * in + j] = 0.5f - float(rand()) / float(RAND_MAX);
            }
        }
}

// Destructor
FLayer::~FLayer() {
    cudaFree(z);
    cudaFree(a);
    cudaFree(bias);
    cudaFree(weight);
    cudaFree(dweight);
    cudaFree(dz);
    cudaFree(da);
}

void FLayer::setOutput0(float *data) { memcpy(a, data, sizeof(float) * outDim); }

void FLayer::clear() {
    memset(a, 0, sizeof(float) * outDim);
    if (inDim)
        memset(z, 0, sizeof(float) * outDim);
}

void FLayer::bp_clear() {
    memset(dweight, 0, sizeof(float) * inDim * outDim);
    memset(da, 0, sizeof(float) * outDim);
    memset(dz, 0, sizeof(float) * outDim);
}


// LeNet

LLayer::LLayer(int M, int N, int O, int arg, double &randTime) {
    this->M = M;
    this->N = N;
    this->O = O;

    if (arg == 0)
        output = Linput_a;
    else if (arg == 1) {
        output = Lc1_a;
        preact = Lc1_z;
        bias = Lc1_bias;
        weight = Lc1_weight;
        d_output = Lc1_da;
        d_preact = Lc1_dz;
        d_weight = Lc1_dweight;
    } else if (arg == 2) {
        output = Ls1_a;
        preact = Ls1_z;
        bias = Ls1_bias;
        weight = Ls1_weight;
        d_output = Ls1_da;
        d_preact = Ls1_dz;
        d_weight = Ls1_dweight;
    } else if (arg == 3) {
        output = Lc2_a;
        preact = Lc2_z;
        bias = Lc2_bias;
        weight = Lc2_weight;
        d_output = Lc2_da;
        d_preact = Lc2_dz;
        d_weight = Lc2_dweight;
    } else if (arg == 4) {
        output = Ls2_a;
        preact = Ls2_z;
        bias = Ls2_bias;
        weight = Ls2_weight;
        d_output = Ls2_da;
        d_preact = Ls2_dz;
        d_weight = Ls2_dweight;
    } else if (arg == 5) {
        output = Lc3_a;
        preact = Lc3_z;
        bias = Lc3_bias;
        weight = Lc3_weight;
        d_output = Lc3_da;
        d_preact = Lc3_dz;
        d_weight = Lc3_dweight;
    } else if (arg == 6) {
        output = Lf1_a;
        preact = Lf1_z;
        bias = Lf1_bias;
        weight = Lf1_weight;
        d_output = Lf1_da;
        d_preact = Lf1_dz;
        d_weight = Lf1_dweight;
    } else if (arg == 7) {
        output = Lf2_a;
        preact = Lf2_z;
        bias = Lf2_bias;
        weight = Lf2_weight;
        d_output = Lf2_da;
        d_preact = Lf2_dz;
        d_weight = Lf2_dweight;
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

LLayer::~LLayer() {
    cudaFree(output);
    cudaFree(preact);
    cudaFree(bias);
    cudaFree(weight);
    cudaFree(d_output);
    cudaFree(d_preact);
    cudaFree(d_weight);
}

void LLayer::setOutput(float *data) { memcpy(output, data, sizeof(float) * O); }

void LLayer::clear() {
    memset(output, 0x00, sizeof(float) * O);
    if (M)
        memset(preact, 0x00, sizeof(float) * O);
}

void LLayer::bp_clear() {
    memset(d_output, 0x00, sizeof(float) * O);
    memset(d_preact, 0x00, sizeof(float) * O);
    memset(d_weight, 0x00, sizeof(float) * M * N);
}

__device__ float Lstep_function(float v) // Sigmoid function::Activation Function
{
    return 1 / (1 + exp(-v));
}

__global__ void Lapply_step_function(float *input, float *output, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] = Lstep_function(input[idx]);
    }
}

__global__ void LmakeError(float *err, float *output, unsigned int Y, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x; // find specific index/thread in GPU
    const int size = blockDim.x * gridDim.x;               // the size of all index/thread in GPU

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
    }
}

__global__ void Lapply_grad(float *output, float *grad, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] += Ldt * grad[idx];
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


// ResNet

// Constructor

RLayer::RLayer(int M, int N, int O, char *arg) {
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
        output = Rinput_a;
    else if (strcmp(arg, "c1") == 0) {
        output = Rc1_a;
        preact = Rc1_z;
        bias = Rc1_bias;
        weight = Rc1_weight;
    } else if (strcmp(arg, "c2") == 0) {
        output = Rc2_a;
        preact = Rc2_z;
        bias = Rc2_bias;
        weight = Rc2_weight;
    } else if (strcmp(arg, "c3") == 0) {
        output = Rc3_a;
        preact = Rc3_z;
        bias = Rc3_bias;
        weight = Rc3_weight;
    } else if (strcmp(arg, "f") == 0) {
        output = Rf_a;
        preact = Rf_z;
        bias = Rf_bias;
        weight = Rf_weight;
    } else if (strcmp(arg, "r") == 0) {
        output = Rr_a;
        preact = Rr_z;
        bias = Rr_bias;
        weight = Rr_weight;
    }

    if (M)
        for (int i = 0; i < N; ++i) {
            bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
            for (int j = 0; j < M; ++j) {
                weight[i * M + j] = 0.5f - float(rand()) / float(RAND_MAX);
            }
        }
}

RLayer::~RLayer() {
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

void RLayer::setOutput(float *data) { memcpy(output, data, sizeof(float) * O); }

// Reset GPU memory between iterations

void RLayer::clear() {
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

__global__ void RmakeError(float *err, float *output, unsigned int Y, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
    }
}

__global__ void Rapply_grad(float *output, float *grad, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] += Rdt * grad[idx];
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