// chenyang's version
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "layer.cu"
#include "../include/mnist.h"

#include <cuda.h>
#include <time.h>

using namespace std;

__global__ void myConv(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT,
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

void myConv_cpu(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT, int IN_N,
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

__global__ void myPooling(float *data, float *res, float *weights, float *biases, int IN_HEIGHT_3, int FILT_HEIGHT,
                          int IN_N, float offset = 1) {
    IN_N *= offset;
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
                   float offset = 0) {
    int START_IN_N = IN_N * offset;
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

__global__ void myFfp(float *data, float *res, float *weight, float *bias, int inDim, int outDim, float offset = 1) {
    outDim *= offset;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outDim) {
        float temp = bias[idx];
        for (int i = 0; i < inDim; i++) {
            temp += weight[idx * inDim + i] * data[i];
        }
        res[idx] = temp;
    }
}
void myFfp_cpu(float *data, float *res, float *weight, float *bias, int inDim, int outDim, float offset = 0) {
    int start_outDim = outDim * offset;
#pragma omp parallel for
    for (int bIdx = start_outDim; bIdx < outDim; bIdx++) {
        float temp = bias[bIdx];
        for (int i = 0; i < inDim; i++) {
            temp += weight[bIdx * inDim + i] * data[i];
        }
        res[bIdx] = temp;
    }
}

bool printTime = true;

static mnist_data *train_set, *test_set;
static unsigned int Ltrain_cnt, Ltest_cnt;

// Define layers of CNN
double mainIniTime = 0, randTime = 0;
double iniStart = gettime();
LLayer l_c3 = LLayer(4 * 4 * 16, 120, 1 * 1 * 120, 5, randTime);
LLayer l_input = LLayer(0, 0, 28 * 28, 0, randTime);
LLayer l_c1 = LLayer(5 * 5, 6, 24 * 24 * 6, 1, randTime);
LLayer l_s1 = LLayer(2 * 2, 1, 12 * 12 * 6, 2, randTime);
LLayer l_c2 = LLayer(5 * 5 * 6, 16, 8 * 8 * 16, 3, randTime);
LLayer l_s2 = LLayer(2 * 2, 1, 4 * 4 * 16, 4, randTime);

LLayer l_f1 = LLayer(120, 84, 84, 6, randTime);
LLayer l_f2 = LLayer(84, 10, 10, 7, randTime);
double iniEnd = gettime();

static void learn(cudaStream_t stream1);
static double forward_pass(double data[28][28], cudaStream_t stream1, float offset = 1);

static inline void loaddata() {
    mnist_load("../data/mnist/train-images.idx3-ubyte", "../data/mnist/train-labels.idx1-ubyte", &train_set, &Ltrain_cnt);
    mnist_load("../data/mnist/t10k-images.idx3-ubyte", "../data/mnist/t10k-labels.idx1-ubyte", &test_set, &Ltest_cnt);
}

inline void get_cuda_size(const int N, int &grid, int &block) {
    int i = -1;
    int temp = N;
    while (temp) {
        temp >>= 1;
        i++;
    }
    block = 1 << int(i / 2);
    grid = ceil(1.0 * N / block);
}

int main(int argc, const char **argv) {
    /////////////////cudastream//////////////////////////
    double mainIniSt = gettime();
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &Linput_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc1_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc1_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc1_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls1_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls1_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls1_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc2_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc2_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc2_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls2_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls2_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ls2_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc3_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc3_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc3_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lc3_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf1_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf1_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf1_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf2_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf2_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Lf2_dz, 0, cudaMemAttachHost);

    cudaMemAdvise(Linput_a, LinputDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lc3_weight, Lc3wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lc3_bias, Lc3N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lc1_weight, Lc1wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lc1_bias, Lc1N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Ls1_weight, Ls1wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Ls1_bias, Ls1N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lc2_weight, Lc2wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lc2_bias, Lc2N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Ls2_weight, Ls2wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Ls2_bias, Ls2N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lf1_weight, Lf1wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lf1_bias, Lf1N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lf2_weight, Lf2wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(Lf2_bias, Lf2N * sizeof(float), cudaMemAdviseSetReadMostly, 0);

    double mainIniEnd = gettime();
    mainIniTime = mainIniEnd - mainIniSt;

    loaddata();
    for (int i = 10; i >= 0; i--) {
        double totalSt = gettime();
        offset = i / 10.0;
        printTime = true;
        learn(stream1);
        double totalEnd = gettime();
        printf("Total time:%lf, rand time:%lf\n", totalEnd - totalSt + mainIniTime + iniEnd - iniStart, randTime);
    }
    return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28], cudaStream_t stream1, float offset) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            Linput_a[i * 28 + j] = data[i][j];
        }
    }
    l_input.clear();
    l_c1.clear();
    l_s1.clear();
    l_c2.clear();
    l_s2.clear();
    l_c3.clear();
    l_f1.clear();
    l_f2.clear();
    cudaEvent_t c1, s1, c2, s2, c3, f1;
    if (cpurun && offset != 1) {
        cudaEventCreate(&c1);
        cudaEventCreate(&s1);
        cudaEventCreate(&c2);
        cudaEventCreate(&s2);
        cudaEventCreate(&c3);
        cudaEventCreate(&f1);
    }
    double start = gettime();
    dim3 gridc1(24, int(6 * offset), 1);
    double t1, t2;
    cout << "Time for kernel: ";
    t1 = gettime();
    if (gpurun && offset)
        myConv<<<gridc1, 24>>>(l_input.output, l_c1.preact, l_c1.weight, l_c1.bias, 28, 5, 1, 6, offset);
    if (cpurun && offset != 1)
        myConv_cpu(l_input.output, l_c1.preact, l_c1.weight, l_c1.bias, 28, 5, 1, 6, offset);
    cudaDeviceSynchronize();
    t2 = gettime();
    // cout << (t2 - t1) * 1000 << "\t";
    if (gpurun) {
        Lapply_step_function<<<64, 64, 0, stream1>>>(l_c1.preact, l_c1.output, l_c1.O);
        if (cpurun && offset != 1)
            cudaEventRecord(c1);
    }

    dim3 grids1(12, int(6 * offset), 1);
    t1 = gettime();
    if (gpurun && offset)
        myPooling<<<grids1, 12>>>(l_c1.output, l_s1.preact, l_s1.weight, l_s1.bias, 24, 2, 6, offset);
    if (cpurun && offset != 1) {
        cudaEventSynchronize(c1);
        myPooling_cpu(l_c1.output, l_s1.preact, l_s1.weight, l_s1.bias, 24, 2, 6, offset);
    }

    cudaDeviceSynchronize();
    t2 = gettime();
    // cout << (t2 - t1) * 1000 << "\t";
    if (gpurun) {
        Lapply_step_function<<<32, 32, 0, stream1>>>(l_s1.preact, l_s1.output, l_s1.O);
        if (cpurun && offset != 1)
            cudaEventRecord(s1);
    }

    dim3 gridc2(8, int(16 * offset), 1);
    t1 = gettime();
    if (gpurun && offset)
        myConv<<<gridc2, 8>>>(l_s1.output, l_c2.preact, l_c2.weight, l_c2.bias, 12, 5, 6, 16, offset);
    if (cpurun && offset != 1) {
        cudaEventSynchronize(s1);
        myConv_cpu(l_s1.output, l_c2.preact, l_c2.weight, l_c2.bias, 12, 5, 6, 16, offset);
    }

    cudaDeviceSynchronize();
    t2 = gettime();
    // cout << (t2 - t1) * 1000 << "\t";
    if (gpurun) {
        Lapply_step_function<<<32, 32, 0, stream1>>>(l_c2.preact, l_c2.output, l_c2.O);
        if (cpurun && offset != 1)
            cudaEventRecord(c2);
    }

    dim3 grids2(4, int(16 * offset), 1);
    t1 = gettime();
    if (gpurun && offset)
        myPooling<<<grids2, 4>>>(l_c2.output, l_s2.preact, l_s2.weight, l_s2.bias, 8, 2, 16, offset);
    if (cpurun && offset != 1) {
        cudaEventSynchronize(c2);
        myPooling_cpu(l_c2.output, l_s2.preact, l_s2.weight, l_s2.bias, 8, 2, 16, offset);
    }
    cudaDeviceSynchronize();
    t2 = gettime();
    // cout << (t2 - t1) * 1000 << "\t";
    if (gpurun) {
        Lapply_step_function<<<16, 16, 0, stream1>>>(l_s2.preact, l_s2.output, l_s2.O);
        if (cpurun && offset != 1)
            cudaEventRecord(s2);
    }
    dim3 gridc3(1, int(120 * offset), 1);
    t1 = gettime();
    if (gpurun && offset)
        myConv<<<gridc3, 1>>>(l_s2.output, l_c3.preact, l_c3.weight, l_c3.bias, 4, 4, 16, 120, offset);
    if (cpurun && offset != 1) {
        cudaEventSynchronize(s2);
        myConv_cpu(l_s2.output, l_c3.preact, l_c3.weight, l_c3.bias, 4, 4, 16, 120, offset);
    }
    cudaDeviceSynchronize();
    t2 = gettime();
    // cout << (t2 - t1) * 1000 << "\t";
    if (gpurun) {
        Lapply_step_function<<<16, 16, 0, stream1>>>(l_c3.preact, l_c3.output, l_c3.O);
        if (cpurun && offset != 1)
            cudaEventRecord(c3);
    }
    int gridf1, blockf1;
    get_cuda_size(int(84 * offset), gridf1, blockf1);
    t1 = gettime();
    if (gpurun && offset)
        myFfp<<<gridf1, blockf1>>>(l_c3.output, l_f1.preact, l_f1.weight, l_f1.bias, 120, 84, offset);
    if (cpurun && offset != 1) {
        cudaEventSynchronize(c3);
        myFfp_cpu(l_c3.output, l_f1.preact, l_f1.weight, l_f1.bias, 120, 84, 0);
    }
    cudaDeviceSynchronize();
    t2 = gettime();
    // cout << (t2 - t1) * 1000 << "\t";
    if (gpurun) {
        Lapply_step_function<<<16, 16, 0, stream1>>>(l_f1.preact, l_f1.output, l_f1.O);
        if (cpurun && offset != 1)
            cudaEventRecord(f1);
    }
    int gridf2, blockf2;
    get_cuda_size(int(10 * offset), gridf2, blockf2);
    t1 = gettime();
    if (gpurun && offset)
        myFfp<<<gridf2, blockf2>>>(l_f1.output, l_f2.preact, l_f2.weight, l_f2.bias, 84, 10, offset);
    if (cpurun && offset != 1) {
        cudaEventSynchronize(f1);
        myFfp_cpu(l_f1.output, l_f2.preact, l_f2.weight, l_f2.bias, 84, 10, 0);
    }
    t2 = gettime();
    if (printTime)
        cout << (t2 - t1) * 1000 << "\t finish\t";
    printTime = false;
    if (gpurun)
        Lapply_step_function<<<4, 4, 0, stream1>>>(l_f2.preact, l_f2.output, l_f2.O);
    cudaDeviceSynchronize();
    double end = gettime();
    return end - start;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24 * 24][5 * 5]) {
    int a = 0;
    (void)unfold_input;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int b = 0;
            for (int x = i; x < i + 2; ++x)
                for (int y = j; y < j + 2; ++y)
                    unfolded[a][b++] = input[x][y];
            a++;
        }
}

static void learn(cudaStream_t stream1) {
    int iter = 1;
    double time_taken = 0.0;
    double start = gettime();
    int round;
    get_round(round);
    while (iter < 0 || iter-- > 0) {
        for (int i = 0; i < 1; ++i) {
            time_taken += forward_pass(train_set[i].data, stream1, offset);
        }
    }
    double end = gettime();
    fprintf(stdout, "offset=%.1f iniTime - %lf, memcpy Time:0, malloc Time:%lf, kernel Time:%lf, ", offset,
            iniEnd - iniStart + mainIniTime, LmallocEnd - LmallocStart, time_taken);
}
