#include "layer.cu"
#include <cuda.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
#include <fstream>

using namespace std;

int readFile(char *fname, float *arr, int n) {
    ifstream ifile;
    ifile.open(fname, ios::in);
    if (!ifile) {
        cerr << "Open File Fail." << endl;
        return 1;
    }
    for (int i = 0; i < n; i++) {
        ifile >> arr[i];
    }
    ifile.close();
    return 0;
}

struct mData {
    float data[InDim];
    float *label;
    mData() { cudaMallocManaged(&label, OutDim * sizeof(float)); }
};

mData train_set[train_cnt];

float testData[2] = {0.6, 0.2};
float testLabel[2] = {0.1, 0.8};

// Define layers
double mainIniTime = 0;
double iniStart = gettime();
static FLayer l_input = FLayer(0, InDim, "input");
static FLayer l_h = FLayer(InDim, hDim, "h");
static FLayer l_f = FLayer(hDim, OutDim, "output");
double iniEnd = gettime();

static double forward_propagation(float *, cudaStream_t);
static void learn(cudaStream_t);
static void getData();

void getData(mData *ds, char *arg = NULL) {
    int dataN;
    if (arg != NULL && strcmp(arg, "test") == 0)
        dataN = test_cnt;
    else
        dataN = train_cnt;
    float arr_in[dataN * InDim];
    readFile("../data/FCNN/input.txt", arr_in, dataN * InDim);
    for (int i = 0; i < dataN; i++) {
        memcpy(ds[i].data, arr_in + i * InDim, InDim * sizeof(float));
    }
}

static void learn(cudaStream_t stream1) {
    float *h_testLabel;
    cudaMalloc(&h_testLabel, sizeof(float) * 2);
    cudaMemcpy(h_testLabel, testLabel, sizeof(float) * 2, cudaMemcpyHostToDevice);

    float err, *tmp_err;
    cudaMallocManaged(&tmp_err, sizeof(float));
    int iter = 1;
    double time_taken = 0.0, total_time_taken = 0.0;
    while (iter-- > 0) {
        time_taken = 0.0;
        err = 0.0f;
        double t1 = gettime();
        for (int i = 0; i < train_cnt; ++i) {
            *tmp_err = 0;
            time_taken += forward_propagation(train_set[i].data, stream1);
        }
        double t2 = gettime();
        err /= train_cnt;
        total_time_taken += t2 - t1;
    }
    fprintf(stdout, "offset=%.1f iniTime - %lf, memcpy Time:0, malloc Time:%lf, kernel Time:%lf, ", offset,
            iniEnd - iniStart + mainIniTime, FmallocEnd - FmallocStart, time_taken);
}

static double forward_propagation(float *data, cudaStream_t stream1) {
    for (int i = 0; i < InDim; i++)
        Finput_a[i] = data[i];
    l_input.clear();
    l_h.clear();
    l_f.clear();
    double start = gettime();
    if (gpurun && offset) {
        fp_z_h<<<int((hDim + 15) / 16), 16, 0, stream1>>>((float *)l_input.a, (float *)l_h.a,
                                                          (float(*)[InDim])l_h.weight, (float *)l_h.bias, offset);
    }

    if (cpurun && offset != 1)
        fp_z_h_cpu((float *)data, (float *)l_h.a, (float(*)[InDim])l_h.weight, (float *)l_h.bias, offset);

    if (offset != 1)
        cudaDeviceSynchronize();

    if (gpurun && offset) {
        fp_z_f<<<int((OutDim + 15) / 16), 16, 0, stream1>>>((float *)l_h.a, l_f.a, (float(*)[hDim])l_f.weight, l_f.bias,
                                                            offset);
    }

    if (cpurun && offset != 1)
        fp_z_f_cpu((float *)l_h.a, l_f.a, (float(*)[hDim])l_f.weight, l_f.bias, offset);

    cudaDeviceSynchronize();
    double end = gettime();
    return ((double)(end - start));
}

int main(int argc, const char **argv) {
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
        return 1;
    }
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &Finput_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Fh_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Fh_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Fh_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Fh_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Fh_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Fh_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Fh_dz, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Foutput_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Foutput_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Foutput_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Foutput_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Foutput_dweight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Foutput_da, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Foutput_dz, 0, cudaMemAttachHost);
    cudaDeviceSynchronize();

    getData(train_set);
    forward_propagation(train_set[0].data, stream1); // to hot up

    for (int i = 10; i >= 0; i--) {
        double total_start = gettime();
        offset = i / 10.0;
        learn(stream1);
        double total_end = gettime();
        printf("total_time:%lf\n", total_end - total_start + mainIniTime + iniEnd - iniStart);
    }
    return 0;
}

////////////////////////// device & global functions ///////////////////////////////////////

__device__ float step_function(float v) { return 1 / (1 + exp(-v)); }

__global__ void apply_step_function(float *input, float *output, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < N)
        output[pos] = step_function(input[pos]);
}

__global__ void makeError(float *dz, float *a, float *Y, const int N, float *err) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < N) {
        dz[pos] = 2 * (a[pos] - Y[pos]) * a[pos] * (1 - a[pos]) / N;
        if (err != NULL)
            atomicAdd(err, (a[pos] - Y[pos]) * (a[pos] - Y[pos]));
    }
}

__global__ void apply_grad(float *output, float *grad, const int N) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < N)
        output[pos] -= Fdt * grad[pos];
}

__global__ void fp_z_h(float *input, float *z, float weight[hDim][InDim], float *bias, float offset) {
    const int bIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int hDim1 = hDim * offset;
    if (bIdx < hDim1) {
        float temp = bias[bIdx];
        for (int i = 0; i < InDim; i++)
            temp += weight[bIdx][i] * input[i];
        z[bIdx] = 1 / (1 + exp(-temp));
    }
}

__global__ void fp_z_f(float *input, float *z, float weight[OutDim][hDim], float *bias, float offset) {
    const int bIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int OutDim1 = OutDim * offset;
    if (bIdx < OutDim1) {
        float temp = bias[bIdx];
        for (int i = 0; i < hDim; i++)
            temp += weight[bIdx][i] * input[i];
        z[bIdx] = 1 / (1 + exp(-temp));
    }
}

/////////////////////////// corun cpu functions ///////////////////////////////////////////////
void apply_step_function_cpu(float *input, float *output, const int N) {
    const int startN = N * offset;

    for (int idx = startN; idx < N; ++idx) {
        output[idx] = sigmoid_cpu(input[idx]);
    }
}

void makeError_cpu(float *dz, float *a, float *Y, const int N, float *err) {
    const int startN = N * 0;
    for (int pos = startN; pos < N; pos++) {
        dz[pos] = 2 * (a[pos] - Y[pos]) * a[pos] * (1 - a[pos]) / N;
        if (err != NULL)
            (*err) += (a[pos] - Y[pos]) * (a[pos] - Y[pos]);
    }
}

void apply_grad_cpu(float *output, float *grad, const int N) {
    const int startN = N * offset;
    for (int idx = startN; idx < N; ++idx) {
        output[idx] -= Fdt * grad[idx];
    }
}

void fp_z_h_cpu(float *input, float *z, float weight[hDim][InDim], float *bias, float offset) {
    const int hDim1 = hDim * offset;
#pragma omp parallel for
    for (int bIdx = hDim1; bIdx < hDim; bIdx++) {
        float temp = bias[bIdx];
        for (int idx = 0; idx < InDim; ++idx) {
            temp += weight[bIdx][idx] * input[idx];
        }
        z[bIdx] = 1 / (1 + exp(-temp));
    }
}

void fp_z_f_cpu(float *input, float *z, float weight[OutDim][hDim], float *bias, float offset) {
    const int OutDim1 = OutDim * offset;
#pragma omp parallel for
    for (int bIdx = OutDim1; bIdx < OutDim; bIdx++) {
        float temp = bias[bIdx];
        for (int idx = 0; idx < hDim; ++idx) {
            temp += weight[bIdx][idx] * input[idx];
        }
        z[bIdx] = 1 / (1 + exp(-temp));
    }
}