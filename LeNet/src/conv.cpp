#ifndef __MY_CONV__
#define __MY_CONV__
#include <bits/stdc++.h>
#include <cuda.h>
#include <omp.h>
#endif

using namespace std;

__global__ void myConv(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT,
                       int IN_N, int OUT_N, float Offset = 1) {
    OUT_N *= Offset;
    int IN_WIDTH_3 = IN_HEIGHT_3, FILT_WIDTH = FILT_HEIGHT;
    int OUT_HEIGHT_3 = IN_HEIGHT_3 - FILT_HEIGHT + 1, OUT_WIDTH_3 = IN_WIDTH_3 - FILT_WIDTH + 1;
    int oh = blockIdx.x;
    int ow = threadIdx.x;
    if (oh >= OUT_HEIGHT_3 || ow >= OUT_WIDTH_3)
        return;
    int ff = blockIdx.y;
    if (ff >= OUT_N)
        return;
    int offset0 = ow + oh * OUT_WIDTH_3 + ff * OUT_WIDTH_3 * OUT_HEIGHT_3;
    float temp = biases[ff];
    for (int cc = 0; cc < IN_N; cc++) {
        for (int fh = 0; fh < FILT_HEIGHT; fh++) {
            for (int fw = 0; fw < FILT_WIDTH; fw++) {
                int index_weight = fw + FILT_WIDTH * (fh + FILT_HEIGHT * (cc + ff * IN_N));
                int index_data = ow + fw + IN_WIDTH_3 * ((oh + fh) + cc * IN_HEIGHT_3);
                temp += data[index_data] * weights[index_weight];
            }
        }
    }
    res[offset0] = temp;
}

void myConv_cpu(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT, int IN_N,
                int OUT_N, float Offset = 0) {
    int START_OUT_N = OUT_N * Offset;
    int IN_WIDTH_3 = IN_HEIGHT_3, FILT_WIDTH = FILT_HEIGHT;
    int OUT_HEIGHT_3 = IN_HEIGHT_3 - FILT_HEIGHT + 1, OUT_WIDTH_3 = IN_WIDTH_3 - FILT_WIDTH + 1;
#pragma omp parallel for
    for (int ff = START_OUT_N; ff < OUT_N; ff++) {

        for (int oh = 0; oh < OUT_WIDTH_3; oh++)
            for (int ow = 0; ow < OUT_HEIGHT_3; ow++) {
                int offset0 = ow + oh * OUT_WIDTH_3 + ff * OUT_WIDTH_3 * OUT_HEIGHT_3;
                float temp = biases[ff];
                for (int cc = 0; cc < IN_N; cc++) {
                    for (int fh = 0; fh < FILT_HEIGHT; fh++) {
                        for (int fw = 0; fw < FILT_WIDTH; fw++) {
                            int index_weight = fw + FILT_WIDTH * (fh + FILT_HEIGHT * (cc + ff * IN_N));
                            int index_data = ow + fw + IN_WIDTH_3 * ((oh + fh) + cc * IN_HEIGHT_3);
                            temp += data[index_data] * weights[index_weight];
                        }
                    }
                }
                res[offset0] = temp;
            }
    }
}

__global__ void myPooling(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT,
                          int IN_N, float Offset = 1) {
    IN_N *= Offset;
    int IN_WIDTH_3 = IN_HEIGHT_3, FILT_WIDTH = FILT_HEIGHT, OUT_N = IN_N;
    int OUT_HEIGHT_3 = IN_HEIGHT_3 / FILT_HEIGHT, OUT_WIDTH_3 = IN_WIDTH_3 / FILT_WIDTH;
    int oh = blockIdx.x;
    int ow = threadIdx.x;
    if (oh >= OUT_HEIGHT_3 || ow >= OUT_WIDTH_3)
        return;
    int ff = blockIdx.y;
    if (ff >= OUT_N)
        return;
    int offset0 = ow + oh * OUT_WIDTH_3 + ff * OUT_WIDTH_3 * OUT_HEIGHT_3;
    float temp = biases[ff];
    for (int fh = 0; fh < FILT_HEIGHT; fh++) {
        for (int fw = 0; fw < FILT_WIDTH; fw++) {
            int index_weight = fw + FILT_WIDTH * fh;
            int index_data = ow * FILT_WIDTH + fw + IN_WIDTH_3 * ((oh * FILT_HEIGHT + fh) + ff * IN_HEIGHT_3);
            temp += data[index_data] * weights[index_weight];
        }
    }
    res[offset0] = temp;
}

void myPooling_cpu(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT, int IN_N,
                   float Offset = 0) {
    int START_IN_N = IN_N * Offset;
    int IN_WIDTH_3 = IN_HEIGHT_3, FILT_WIDTH = FILT_HEIGHT;
    int OUT_HEIGHT_3 = IN_HEIGHT_3 / FILT_HEIGHT, OUT_WIDTH_3 = IN_WIDTH_3 / FILT_WIDTH;
#pragma omp parallel for
    for (int ff = START_IN_N; ff < IN_N; ff++)
        for (int oh = 0; oh < OUT_HEIGHT_3; oh++)
            for (int ow = 0; ow < OUT_WIDTH_3; ow++) {
                int offset0 = ow + oh * OUT_WIDTH_3 + ff * OUT_WIDTH_3 * OUT_HEIGHT_3;
                float temp = biases[ff];
                for (int fh = 0; fh < FILT_HEIGHT; fh++) {
                    for (int fw = 0; fw < FILT_WIDTH; fw++) {
                        int index_weight = fw + FILT_WIDTH * fh;
                        int index_data =
                            ow * FILT_WIDTH + fw + IN_WIDTH_3 * ((oh * FILT_HEIGHT + fh) + ff * IN_HEIGHT_3);
                        temp += data[index_data] * weights[index_weight];
                    }
                }
                res[offset0] = temp;
            }
}

__global__ void myFfp(float *data, float *res, float *weight, float *bias, int inDim, int outDim, float Offset = 1) {
    outDim *= Offset;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outDim) {
        float temp = bias[idx];
        for (int i = 0; i < inDim; i++) {
            temp += weight[idx * inDim + i] * data[i];
        }
        res[idx] = temp;
    }
}
void myFfp_cpu(float *data, float *res, float *weight, float *bias, int inDim, int outDim, float Offset = 0) {
    int start_outDim = outDim * Offset;
#pragma omp parallel for
    for (int bIdx = start_outDim; bIdx < outDim; bIdx++) {
        float temp = bias[bIdx];
        for (int i = 0; i < inDim; i++) {
            temp += weight[bIdx * inDim + i] * data[i];
        }
        res[bIdx] = temp;
    }
}