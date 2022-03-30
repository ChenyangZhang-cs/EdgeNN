// chenyang's version
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "conv.cpp"
#include "layer.cu"
#include "mnist.h"

#include <cuda.h>
#include <time.h>

using namespace std;

bool printTime = true;

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
double mainIniTime = 0, randTime = 0;
double iniStart = gettime();
Layer l_c3 = Layer(4 * 4 * 16, 120, 1 * 1 * 120, 5, randTime);
Layer l_input = Layer(0, 0, 28 * 28, 0, randTime);
Layer l_c1 = Layer(5 * 5, 6, 24 * 24 * 6, 1, randTime);
Layer l_s1 = Layer(2 * 2, 1, 12 * 12 * 6, 2, randTime);
Layer l_c2 = Layer(5 * 5 * 6, 16, 8 * 8 * 16, 3, randTime);
Layer l_s2 = Layer(2 * 2, 1, 4 * 4 * 16, 4, randTime);

Layer l_f1 = Layer(120, 84, 84, 6, randTime);
Layer l_f2 = Layer(84, 10, 10, 7, randTime);
double iniEnd = gettime();

static void learn(cudaStream_t stream1);
static double forward_pass(double data[28][28], cudaStream_t stream1, float offset = 1);

static inline void loaddata() {
    mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &train_set, &train_cnt);
    mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt);
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
    cudaStreamAttachMemAsync(stream1, &input_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s1_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s1_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s1_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s2_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s2_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &s2_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f1_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f1_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f1_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f2_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f2_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f2_dz, 0, cudaMemAttachHost);

    cudaMemAdvise(input_a, inputDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(c3_weight, c3wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(c3_bias, c3N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(c1_weight, c1wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(c1_bias, c1N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(s1_weight, s1wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(s1_bias, s1N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(c2_weight, c2wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(c2_bias, c2N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(s2_weight, s2wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(s2_bias, s2N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(f1_weight, f1wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(f1_bias, f1N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(f2_weight, f2wDim * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(f2_bias, f2N * sizeof(float), cudaMemAdviseSetReadMostly, 0);

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
            input_a[i * 28 + j] = data[i][j];
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
        apply_step_function<<<64, 64, 0, stream1>>>(l_c1.preact, l_c1.output, l_c1.O);
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
        apply_step_function<<<32, 32, 0, stream1>>>(l_s1.preact, l_s1.output, l_s1.O);
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
        apply_step_function<<<32, 32, 0, stream1>>>(l_c2.preact, l_c2.output, l_c2.O);
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
        apply_step_function<<<16, 16, 0, stream1>>>(l_s2.preact, l_s2.output, l_s2.O);
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
        apply_step_function<<<16, 16, 0, stream1>>>(l_c3.preact, l_c3.output, l_c3.O);
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
        apply_step_function<<<16, 16, 0, stream1>>>(l_f1.preact, l_f1.output, l_f1.O);
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
        apply_step_function<<<4, 4, 0, stream1>>>(l_f2.preact, l_f2.output, l_f2.O);
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
            iniEnd - iniStart + mainIniTime, mallocEnd - mallocStart, time_taken);
}
