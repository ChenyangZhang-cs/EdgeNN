#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "layer.cu"
#include "mnist.h"
#include <cstdio>
#include <cuda.h>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
double iniStart = gettime();
static Layer l_input = Layer(0, 0, 28 * 28, "input");
static Layer l_c1 = Layer(5 * 5, 6, 24 * 24 * 6, "c1");
static Layer l_c2 = Layer(2 * 2, 6, 12 * 12 * 6, "c2");
static Layer l_c3 = Layer(2 * 2, 6, 6 * 6 * 6, "c3");
static Layer l_f = Layer(6 * 6 * 6, 10, 10, "f");
static Layer l_r = Layer(4 * 4, 1, 6 * 6 * 6, "r");
double iniEnd = gettime();
static void learn();
static double forward_pass(double data[28][28]);

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
    srand(time(NULL));
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
        return 1;
    }
    //////////////////////////////cudastream////////////////////////////////////////
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &input_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &c3_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &f_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &r_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &r_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &r_z, 0, cudaMemAttachHost);

    loaddata();
    for (int i = 10; i >= 0; i--) {
        double totalSt = gettime();
        offset = i / 10.0;
        learn();
        double totalEnd = gettime();
        printf("Total Time:%lf\n", totalEnd - totalSt);
    }
    return 0;
}

// Forward propagation of a single row in dataset

static double forward_pass(double data[28][28]) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input_a[i * 28 + j] = data[i][j];
        }
    }
    double start = gettime();

    if (offset) {
        int c1_grid, c1_block;
        get_cuda_size(int(6 * 24 * 24 * offset), c1_grid, c1_block);
        fp_preact_c1<<<c1_grid, c1_block>>>((float(*)[28])l_input.output, (float(*)[24][24])l_c1.output,
                                            (float(*)[5][5])l_c1.weight, l_c1.bias, offset);
    }
    if (offset != 1)
        fp_preact_c1_cpu((float(*)[28])l_input.output, (float(*)[24][24])l_c1.output, (float(*)[5][5])l_c1.weight,
                         l_c1.bias);

    if (offset != 1)
        cudaDeviceSynchronize();

    if (offset) {
        int r_grid, r_block;
        get_cuda_size(int(6 * 6 * 6 * offset), r_grid, r_block);
        fp_preact_r<<<r_grid, r_block>>>((float(*)[24][24])l_c1.output, (float(*)[6][6])l_r.preact,
                                         (float(*)[4][4])l_r.weight, *l_r.bias, offset);
    }
    if (offset != 1)
        fp_preact_r_cpu((float(*)[24][24])l_c1.output, (float(*)[6][6])l_r.preact, (float(*)[4][4])l_r.weight,
                        *l_r.bias);
    if (offset != 1)
        cudaDeviceSynchronize();

    if (offset) {
        int c2_grid, c2_block;
        get_cuda_size(int(6 * 12 * 12 * offset), c2_grid, c2_block);
        fp_preact_c2<<<c2_grid, c2_block>>>((float(*)[24][24])l_c1.output, (float(*)[12][12])l_c2.output,
                                            (float(*)[2][2])l_c2.weight, l_c2.bias, offset);
    }
    if (offset != 1)
        fp_preact_c2_cpu((float(*)[24][24])l_c1.output, (float(*)[12][12])l_c2.output, (float(*)[2][2])l_c2.weight,
                         l_c2.bias);

    if (offset != 1)
        cudaDeviceSynchronize();

    if (offset) {
        int c3_grid, c3_block;
        get_cuda_size(int(6 * 6 * 6 * offset), c3_grid, c3_block);
        fp_preact_c3<<<c3_grid, c3_block>>>((float(*)[12][12])l_c2.output, (float(*)[6][6])l_c3.preact,
                                            (float(*)[2][2])l_c3.weight, l_c3.bias, offset);
    }
    if (offset != 1)
        fp_preact_c3_cpu((float(*)[12][12])l_c2.output, (float(*)[6][6])l_c3.preact, (float(*)[2][2])l_c3.weight,
                         l_c3.bias);

    int add_grid, add_block;
    get_cuda_size(6 * 6 * 6, add_grid, add_block);
    fp_add_res<<<add_grid, add_block>>>((float(*)[6][6])l_c3.preact, (float(*)[6][6])l_r.preact);
    apply_sigmoid<<<128, 128>>>(l_c3.preact, l_c3.output, l_c3.O);

    if (offset != 1)
        cudaDeviceSynchronize();

    if (offset) {
        int f_grid, f_block;
        get_cuda_size(int(10 * offset), f_grid, f_block);
        fp_preact_f<<<f_grid, f_block>>>((float(*)[6][6])l_c3.output, l_f.output, (float(*)[6][6][6])l_f.weight,
                                         l_f.bias, offset);
    }
    if (offset != 1)
        fp_preact_f_cpu((float(*)[6][6])l_c3.output, l_f.output, (float(*)[6][6][6])l_f.weight, l_f.bias);

    cudaDeviceSynchronize();

    double end = gettime();

    return ((double)(end - start));
}

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

static void learn() {
    int iter = 1;
    double time_taken = 0.0;
    while (iter < 0 || iter-- > 0) {
        for (int i = 0; i < 500; ++i) {
            time_taken += forward_pass(train_set[i].data);
        }
    }
    fprintf(stdout, "offset=%.1f iniTime - %lf, memcpy Time:0, malloc Time:%lf, kernel Time:%lf,", offset,
            iniEnd - iniStart, mallocEnd - mallocStart, time_taken);
}
