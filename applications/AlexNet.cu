#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "layer.cu"
#include "../include/mnist.h"
#include "../include/pixels.h"

#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <time.h>

using namespace std;

// Define layers of CNN
double iniStart = gettime();
ALayer L_input = ALayer(0, 0, 227 * 227 * 3, "input");
ALayer L_c1 = ALayer(11 * 11 * 3, 2 * 48, 2 * 55 * 55 * 48, "c1");
ALayer L_p1 = ALayer(3 * 3, 2 * 1, 2 * 31 * 31 * 48, "p1");
ALayer L_c2 = ALayer(5 * 5 * 48, 2 * 128, 2 * 128 * 27 * 27, "c2");
ALayer L_p2 = ALayer(3 * 3, 2 * 1, 2 * 15 * 15 * 128, "p2");
ALayer L_c3 = ALayer(3 * 3 * 256, 384, 2 * 13 * 13 * 192, "c3");
ALayer L_c4 = ALayer(3 * 3 * 384, 2 * 192, 2 * 13 * 13 * 192, "c4");
ALayer L_c5 = ALayer(3 * 3 * 384, 2 * 128, 2 * 13 * 13 * 128, "c5");
ALayer L_p3 = ALayer(3 * 3, 2 * 1, 2 * 6 * 6 * 128, "p3");
ALayer L_f1 = ALayer(6 * 6 * 256, 2 * 2048, 4096 * 1, "f1");
ALayer L_f2 = ALayer(1 * 4096, 2 * 2048, 4096 * 1, "f2");
ALayer L_f3 = ALayer(1 * 4096, 1000, 1000, "f3");
double iniEnd = gettime();

static void learn(double data[227][227][3], cudaStream_t stream1);
static double forward_pass(double data[227][227][3], cudaStream_t stream1);

int main(int argc, const char **argv) {
    srand(time(NULL));
    double test_data[227][227][3] = {0.0};
    for (int i = 0; i < 227; i++) {
        for (int j = 0; j < 227; j++) {
            for (int k = 0; k < 3; k++) {
                test_data[j][k][i] = double(PIXELS[j][i][k]);
            }
        }
    }
    /////////////////cudastream//////////////////////////
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &Ainput_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_o, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_o, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_z, 0, cudaMemAttachHost);

    forward_pass(test_data, stream1); // to hot up
    for (int i = 10; i >= 0; i--) {
        double totalSt = gettime();
        offset = i / 10.0;
        learn(test_data, stream1);
        double totalEnd = gettime();
        printf("Total Time:%lf\n", totalEnd - totalSt + iniEnd - iniStart);
    }
    return 0;
}

static double forward_pass(double data[227][227][3], cudaStream_t stream1) {
    for (int i = 0; i < 227; ++i) {
        for (int j = 0; j < 227; ++j) {
            for (int k = 0; k < 3; k++) {
                Ainput_a[i * 227 * 3 + j * 3 + k] = data[i][j][k];
            }
        }
    }

    cudaEvent_t p1, p2, p3, f1, f2, c3, c4;
    if (offset != 1) {
        cudaEventCreate(&p1);
        cudaEventCreate(&p2);
        cudaEventCreate(&p3);
        cudaEventCreate(&f1);
        cudaEventCreate(&f2);
        cudaEventCreate(&c3);
        cudaEventCreate(&c4);
    }
    double start = gettime();
    dim3 Bc1(55, 55);
    if (offset)
        fp_preact_c1<<<int(96 * offset), Bc1, 0, stream1>>>((float(*)[227][3])L_input.output,
                                                            (float(*)[55][55])L_c1.preact,
                                                            (float(*)[11][11][3])L_c1.weight, L_c1.bias);
    if (offset != 1)
        fp_preact_c1_cpu((float(*)[227][3])L_input.output, (float(*)[55][55])L_c1.preact,
                         (float(*)[11][11][3])L_c1.weight, L_c1.bias);
    Aapply_step_function<<<112, 1280, 0, stream1>>>(L_c1.preact, L_c1.act_result, L_c1.O);
    normalization_function<<<112, 640, 0, stream1>>>(L_c1.act_result, L_c1.output, L_c1.O, L_c1.N);
    dim3 ft_map(27, 27);
    fp_preact_p1<<<96, ft_map, 0, stream1>>>((float(*)[55][55])L_c1.output, (float(*)[31][31])L_p1.output);
    if (offset != 1)
        cudaEventRecord(p1);
    dim3 Bc2(27, 27);
    if (offset)
        fp_preact_c2<<<int(256 * offset), Bc2, 0, stream1>>>(
            (float(*)[31][31])L_p1.output, (float(*)[27][27])L_c2.preact, (float(*)[96][5][5])L_c2.weight, L_c2.bias);
    if (offset != 1) {
        cudaEventSynchronize(p1);
        fp_preact_c2_cpu((float(*)[31][31])L_p1.output, (float(*)[27][27])L_c2.preact, (float(*)[96][5][5])L_c2.weight,
                         L_c2.bias);
    }
    Aapply_step_function<<<112, 1280, 0, stream1>>>(L_c2.preact, L_c2.act_result, L_c2.O);
    normalization_function<<<112, 1280, 0, stream1>>>(L_c2.act_result, L_c2.output, L_c2.O, L_c1.N);
    dim3 ft_map1(13, 13);
    fp_preact_p2<<<256, ft_map1, 0, stream1>>>((float(*)[27][27])L_c2.output, (float(*)[15][15])L_p2.output);
    if (offset != 1)
        cudaEventRecord(p2);
    dim3 Bc3(13, 13);
    if (offset)
        fp_preact_c3<<<int(384 * offset), Bc3, 0, stream1>>>(
            (float(*)[15][15])L_p2.output, (float(*)[13][13])L_c3.preact, (float(*)[256][3][3])L_c3.weight, L_c3.bias);
    if (offset != 1) {
        cudaEventSynchronize(p2);
        fp_preact_c3_cpu((float(*)[15][15])L_p2.output, (float(*)[13][13])L_c3.preact, (float(*)[256][3][3])L_c3.weight,
                         L_c3.bias);
    }
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c3.preact, L_c3.output, L_c3.O);
    if (offset != 1)
        cudaEventRecord(c3);
    dim3 Bc4(12, 12);
    if (offset)
        fp_preact_c4<<<int(384 * offset), Bc4, 0, stream1>>>(
            (float(*)[13][13])L_c3.output, (float(*)[13][13])L_c4.preact, (float(*)[384][3][3])L_c4.weight, L_c4.bias);
    if (offset != 1) {
        cudaEventSynchronize(c3);
        fp_preact_c4_cpu((float(*)[13][13])L_c3.output, (float(*)[13][13])L_c4.preact, (float(*)[384][3][3])L_c4.weight,
                         L_c4.bias);
    }
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c4.preact, L_c4.output, L_c4.O);
    if (offset != 1)
        cudaEventRecord(c4);

    dim3 Bc5(12, 12);
    if (offset)
        fp_preact_c5<<<int(256 * offset), Bc5, 0, stream1>>>(
            (float(*)[13][13])L_c4.output, (float(*)[13][13])L_c5.preact, (float(*)[384][3][3])L_c5.weight, L_c5.bias);
    if (offset != 1) {
        cudaEventSynchronize(c4);
        fp_preact_c5_cpu((float(*)[13][13])L_c4.output, (float(*)[13][13])L_c5.preact, (float(*)[384][3][3])L_c5.weight,
                         L_c5.bias);
    }
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c5.preact, L_c5.output, L_c5.O);
    dim3 ft_map2(6, 6);
    fp_preact_p3<<<256, ft_map2, 0, stream1>>>((float(*)[13][13])L_c5.output, (float(*)[6][6])L_p3.output);
    if (offset != 1)
        cudaEventRecord(p3);
    if (offset)
        fp_preact_f1<<<int(4096 * offset), 1, 0, stream1>>>((float(*)[6][6])L_p3.output, L_f1.preact,
                                                            (float(*)[256][6][6])L_f1.weight, L_f1.bias);
    if (offset != 1) {
        cudaEventSynchronize(p3);
        fp_preact_f1_cpu((float(*)[6][6])L_p3.output, L_f1.preact, (float(*)[256][6][6])L_f1.weight, L_f1.bias);
    }
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_f1.preact, L_f1.output, L_f1.O);
    if (offset != 1)
        cudaEventRecord(f1);
    if (offset)
        fp_preact_f2<<<int(4096 * offset), 1, 0, stream1>>>(L_f1.output, L_f2.preact, (float(*)[4096])L_f2.weight,
                                                            L_f2.bias);
    if (offset != 1) {
        cudaEventSynchronize(f1);
        fp_preact_f2_cpu(L_f1.output, L_f2.preact, (float(*)[4096])L_f2.weight, L_f2.bias);
    }
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_f2.preact, L_f2.output, L_f2.O);
    if (offset != 1)
        cudaEventRecord(f2);
    if (offset)
        fp_preact_f3<<<int(1000 * offset), 1, 0, stream1>>>(L_f2.output, L_f3.preact, (float(*)[4096])L_f3.weight,
                                                            L_f3.bias);
    if (offset != 1) {
        cudaEventSynchronize(f2);
        fp_preact_f3_cpu(L_f2.output, L_f3.preact, (float(*)[4096])L_f3.weight, L_f3.bias);
    }
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_f3.preact, L_f3.output, L_f3.O);
    cudaDeviceSynchronize();
    double end = gettime();
    return ((double)(end - start));
}

static void learn(double data[227][227][3], cudaStream_t stream1) {
    fflush(stdout);
    int iter = 1;
    double time_taken = 0.0;
    while (iter < 0 || iter-- > 0) {
        for (int i = 0; i < 5; ++i) {
            time_taken += forward_pass(data, stream1);
        }
    }
    fprintf(stdout, "offset=%.1f iniTime - %lf, memcpy Time:0, malloc Time:%lf, kernel Time:%lf,", offset,
            iniEnd - iniStart, mallocEnd - mallocStart, time_taken);
}
