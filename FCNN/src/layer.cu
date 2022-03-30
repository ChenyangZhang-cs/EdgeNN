#include "layer.h"

mLayer::mLayer(int in, int out, char *arg) {
    this->inDim = in;
    this->outDim = out;

    if (arg != NULL && strcmp(arg, "input") == 0)
        a = input_a;

    if (arg != NULL && (strcmp(arg, "h") == 0 || strcmp(arg, "htest") == 0)) {
        weight = h_weight;
        bias = h_bias;
        a = h_a;
        z = h_z;
        dweight = h_dweight;
        da = h_da;
        dz = h_dz;
    }

    if (arg != NULL && (strcmp(arg, "output") == 0 || strcmp(arg, "outputtest") == 0)) {
        weight = output_weight;
        bias = output_bias;
        a = output_a;
        z = output_z;
        dweight = output_dweight;
        da = output_da;
        dz = output_dz;
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
mLayer::~mLayer() {
    cudaFree(z);
    cudaFree(a);
    cudaFree(bias);
    cudaFree(weight);
    cudaFree(dweight);
    cudaFree(dz);
    cudaFree(da);
}

void mLayer::setOutput0(float *data) { memcpy(a, data, sizeof(float) * outDim); }

void mLayer::clear() {
    memset(a, 0, sizeof(float) * outDim);
    if (inDim)
        memset(z, 0, sizeof(float) * outDim);
}

void mLayer::bp_clear() {
    memset(dweight, 0, sizeof(float) * inDim * outDim);
    memset(da, 0, sizeof(float) * outDim);
    memset(dz, 0, sizeof(float) * outDim);
}