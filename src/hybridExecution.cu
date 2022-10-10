/* This code provides an example about how to implement the two memory usage strategies.*/

#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include "../src/layer.cu"

__global__ void conv_gpu(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT,
                       int IN_N, int OUT_N, float offset = 1) {
    OUT_N *= offset;
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

void conv_cpu(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT, int IN_N,
                int OUT_N, float offset = 0) {
    int START_OUT_N = OUT_N * offset;
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

int main(){
    double randTime = 0.0;
    LLayer l_input = LLayer(0, 0, 28 * 28, 0, randTime);
    LLayer l_c1 = LLayer(5 * 5, 6, 24 * 24 * 6, 1, randTime);

    float offset = 0.5;
    if (offset != 0){
        dim3 gridc1(24, int(6 * offset), 1);
        conv_gpu<<<gridc1, 24>>>(l_input.output, l_c1.preact, l_c1.weight, l_c1.bias, 28, 5, 1, 6, offset);
    }
    if (offset != 1)
        conv_cpu(l_input.output, l_c1.preact, l_c1.weight, l_c1.bias, 28, 5, 1, 6, offset);
    cudaDeviceSynchronize();
    
    return 0;
}