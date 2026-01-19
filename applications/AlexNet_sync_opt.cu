#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include "../src/layer.cu"
#include "../include/mnist.h"
#include "../include/pixels.h"

#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <time.h>
#include <future>

using namespace std;

#define dout if(false) cout

__device__ inline void wait_flag(volatile int* global_flag) {
    // while (*global_flag == 0);
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        while (*global_flag == 0);
    }
    
    __syncthreads();
}

__global__ void Aapply_step_function_flag(volatile int* flag, float *input, float *output, const int N) {
    wait_flag(flag);
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx) {
        output[idx] = Astep_function(input[idx]);
    }
}

__global__ void normalization_function_flag(volatile int* flag, float *input, float *output, const int O, const int N) {
    wait_flag(flag);
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    for (int idx = O * pos / size; idx < O * (pos + 1) / size; ++idx) {
        output[idx] = normalization(input, input[idx], idx, O, N);
    }
}

__global__ void fp_preact_c1_flag(volatile int* flag, float input[227][227][3], float preact[96][55][55], float weight[96][11][11][3],
                             float bias[96]) {
    wait_flag(flag);
    int i3 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i3];
    for (int i2 = 0; i2 < 11; i2++)
        for (int i1 = 0; i1 < 11; i1++)
            for (int i6 = 0; i6 < 3; i6++) {
                temp += weight[i3][i1][i2][i6] * input[i4 * 4 + i1][i5 * 4 + i2][i6];
            }
    preact[i3][i4][i5] = temp;
}

__global__ void fp_preact_p1_flag(volatile int* flag, float input[96][55][55], float preact[96][31][31]) {
    wait_flag(flag);
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

__global__ void fp_preact_c2_flag(volatile int* flag, float input[96][31][31], float preact[256][27][27], float weight[256][96][5][5],
                             float bias[256]) {
    wait_flag(flag);
    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 96; i3++)
        for (int i2 = 0; i2 < 5; i2++)
            for (int i1 = 0; i1 < 5; i1++)
                temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
    preact[i6][i4][i5] = temp;
}

__global__ void fp_preact_p2_flag(volatile int* flag, float input[256][27][27], float preact[256][15][15]) {
    wait_flag(flag);
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

__global__ void fp_preact_c3_flag(volatile int* flag, float input[256][15][15], float preact[384][13][13], float weight[384][256][3][3],
                             float bias[384]) {
    wait_flag(flag);
    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 256; i3++)
        for (int i2 = 0; i2 < 3; i2++)
            for (int i1 = 0; i1 < 3; i1++) {
                temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
            }
    preact[i6][i4][i5] = temp;
}

__global__ void fp_preact_c4_flag(volatile int* flag, float input[384][13][13], float preact[384][13][13], float weight[384][384][3][3],
                             float bias[384]) {
    wait_flag(flag);
    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 384; i3++)
        for (int i2 = 0; i2 < 3; i2++)
            for (int i1 = 0; i1 < 3; i1++) {
                temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
            }
    preact[i6][i4 + 1][i5 + 1] = temp;
}

__global__ void fp_preact_c5_flag(volatile int* flag, float input[384][13][13], float preact[256][13][13], float weight[256][384][3][3],
                             float bias[256]) {
    wait_flag(flag);
    int i6 = blockIdx.x, i4 = threadIdx.y, i5 = threadIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 384; i3++)
        for (int i2 = 0; i2 < 3; i2++)
            for (int i1 = 0; i1 < 3; i1++) {
                temp += weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2];
            }
    preact[i6][i4 + 1][i5 + 1] = temp;
}

__global__ void fp_preact_p3_flag(volatile int* flag, float input[256][13][13], float preact[256][6][6]) {
    wait_flag(flag);
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

__global__ void fp_preact_f1_flag(volatile int* flag, float input[256][6][6], float preact[4096], float weight[4096][256][6][6],
                             float bias[4096]) {
    wait_flag(flag);
    int i6 = blockIdx.x;
    float temp = bias[i6];

    for (int i3 = 0; i3 < 256; i3++)
        for (int i2 = 0; i2 < 6; i2++)
            for (int i1 = 0; i1 < 6; i1++) {
                temp += weight[i6][i3][i1][i2] * input[i3][i1][i2];
            }
    preact[i6] = temp;
}

__global__ void fp_preact_f2_flag(volatile int* flag, float input[4096], float preact[4096], float weight[4096][4096], float bias[4096]) {
    wait_flag(flag);
    int i1 = blockIdx.x;
    float temp = bias[i1];
    for (int i2 = 0; i2 < 4096; i2++) {
        temp += weight[i1][i2] * input[i2];
    }
    preact[i1] = temp;
}

__global__ void fp_preact_f3_flag(volatile int* flag, float input[4096], float preact[1000], float weight[1000][4096], float bias[1000]) {
    wait_flag(flag);
    int i1 = blockIdx.x;
    float temp = bias[i1];
    for (int i2 = 0; i2 < 4096; i2++) {
        temp += weight[i1][i2] * input[i2];
    }
    preact[i1] = temp;
}

class PersistentThread {
public:
    PersistentThread() : m_stop(false) {
        m_thread = std::thread(&PersistentThread::worker_loop, this);
    }

    ~PersistentThread() {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_one();
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }

    template<typename F>
    std::future<void> submit(F&& func) {
        auto promise_ptr = std::make_shared<std::promise<void>>();
        std::future<void> future = promise_ptr->get_future();

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_stop) {
                throw std::runtime_error("submit on stopped PersistentThread");
            }

            m_tasks.push(
                [func = std::forward<F>(func), promise_ptr]() mutable {
                    try {
                        func();
                        promise_ptr->set_value();
                    } catch (...) {
                        try {
                            promise_ptr->set_exception(std::current_exception());
                        } catch (...) {}
                    }
                }
            );
        }

        m_cv.notify_one();

        return future;
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this] {
                    return m_stop || !m_tasks.empty();
                });

                if (m_stop && m_tasks.empty()) {
                    return;
                }

                task = std::move(m_tasks.front());
                m_tasks.pop();
            
            }

            task();
        }
    }

    std::thread m_thread;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stop;
};

// Persistent Thread and Flags
PersistentThread gpu_pt;
volatile int *g_h_flags = NULL;
int *g_d_flags = NULL;
const int NUM_FLAGS = 10;

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

    // Allocate flags in mapped memory
    cudaHostAlloc(&g_h_flags, sizeof(int) * NUM_FLAGS, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&g_d_flags, (void*)g_h_flags, 0);

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
    
    cudaFreeHost((void*)g_h_flags);
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

    // Reset flags
    for(int i=0; i<NUM_FLAGS; ++i) g_h_flags[i] = 0;

    // Use volatile flags for execution
    volatile int* d_flag_c1 = &g_d_flags[0];
    volatile int* d_flag_ac1 = &g_d_flags[1];
    volatile int* d_flag_ac2 = &g_d_flags[2];
    volatile int* d_flag_ac3 = &g_d_flags[3];
    volatile int* d_flag_ac4 = &g_d_flags[4];
    volatile int* d_flag_ac5 = &g_d_flags[5];
    volatile int* d_flag_f1 = &g_d_flags[6];
    volatile int* d_flag_f2 = &g_d_flags[7];
    volatile int* d_flag_f3 = &g_d_flags[8];

    // Host flags to set
    volatile int* h_flag_c1 = &g_h_flags[0];
    volatile int* h_flag_ac1 = &g_h_flags[1];
    volatile int* h_flag_ac2 = &g_h_flags[2];
    volatile int* h_flag_ac3 = &g_h_flags[3];
    volatile int* h_flag_ac4 = &g_h_flags[4];
    volatile int* h_flag_ac5 = &g_h_flags[5];
    volatile int* h_flag_f1 = &g_h_flags[6];
    volatile int* h_flag_f2 = &g_h_flags[7];
    volatile int* h_flag_f3 = &g_h_flags[8];

    cudaEvent_t p1, p2, p3, f1, f2, c3, c4;
    // Only create events if needed
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

    *h_flag_c1 = 1;

    // ----------------------------------------------------------------
    // GPU
    // ----------------------------------------------------------------
    auto gpu_thread = gpu_pt.submit([=]() mutable {
        // Layer 1: C1 -> P1
        dim3 Bc1(55, 55);
        if (offset) {
            fp_preact_c1_flag<<<int(96 * offset), Bc1, 0, stream1>>>(d_flag_c1, 
                (float(*)[227][3])L_input.output, (float(*)[55][55])L_c1.preact,
                (float(*)[11][11][3])L_c1.weight, L_c1.bias);
        }
        Aapply_step_function_flag<<<112, 1280, 0, stream1>>>(d_flag_ac1, L_c1.preact, L_c1.act_result, L_c1.O);
        normalization_function_flag<<<112, 640, 0, stream1>>>(d_flag_ac1, L_c1.act_result, L_c1.output, L_c1.O, L_c1.N);
        dim3 ft_map(27, 27);
        fp_preact_p1_flag<<<96, ft_map, 0, stream1>>>(d_flag_ac1, (float(*)[55][55])L_c1.output, (float(*)[31][31])L_p1.output);
        if (offset != 1) cudaEventRecord(p1, stream1);

        // Layer 2: C2 -> P2
        dim3 Bc2(27, 27);
        if (offset)
            fp_preact_c2_flag<<<int(256 * offset), Bc2, 0, stream1>>>(d_flag_ac1, 
                (float(*)[31][31])L_p1.output, (float(*)[27][27])L_c2.preact, (float(*)[96][5][5])L_c2.weight, L_c2.bias);

        Aapply_step_function_flag<<<112, 1280, 0, stream1>>>(d_flag_ac2, L_c2.preact, L_c2.act_result, L_c2.O);
        normalization_function_flag<<<112, 1280, 0, stream1>>>(d_flag_ac2, L_c2.act_result, L_c2.output, L_c2.O, L_c1.N);
        dim3 ft_map1(13, 13);
        fp_preact_p2_flag<<<256, ft_map1, 0, stream1>>>(d_flag_ac2, (float(*)[27][27])L_c2.output, (float(*)[15][15])L_p2.output);
        if (offset != 1) cudaEventRecord(p2, stream1);

        // Layer 3: C3
        dim3 Bc3(13, 13);
        if (offset)
            fp_preact_c3_flag<<<int(384 * offset), Bc3, 0, stream1>>>(d_flag_ac2, 
                (float(*)[15][15])L_p2.output, (float(*)[13][13])L_c3.preact, (float(*)[256][3][3])L_c3.weight, L_c3.bias);

        Aapply_step_function_flag<<<128, 128, 0, stream1>>>(d_flag_ac3, L_c3.preact, L_c3.output, L_c3.O);
        if (offset != 1) cudaEventRecord(c3, stream1);
        
        // Layer 4: C4
        dim3 Bc4(12, 12);
        if (offset)
            fp_preact_c4_flag<<<int(384 * offset), Bc4, 0, stream1>>>(d_flag_ac3, 
                (float(*)[13][13])L_c3.output, (float(*)[13][13])L_c4.preact, (float(*)[384][3][3])L_c4.weight, L_c4.bias);

        Aapply_step_function_flag<<<128, 128, 0, stream1>>>(d_flag_ac4, L_c4.preact, L_c4.output, L_c4.O);
        if (offset != 1) cudaEventRecord(c4, stream1);

        // Layer 5: C5 -> P3
        dim3 Bc5(12, 12);
        if (offset)
            fp_preact_c5_flag<<<int(256 * offset), Bc5, 0, stream1>>>(d_flag_ac4, 
                (float(*)[13][13])L_c4.output, (float(*)[13][13])L_c5.preact, (float(*)[384][3][3])L_c5.weight, L_c5.bias);

        Aapply_step_function_flag<<<128, 128, 0, stream1>>>(d_flag_ac5, L_c5.preact, L_c5.output, L_c5.O);
        dim3 ft_map2(6, 6);
        fp_preact_p3_flag<<<256, ft_map2, 0, stream1>>>(d_flag_ac5, (float(*)[13][13])L_c5.output, (float(*)[6][6])L_p3.output);
        
        if (offset != 1) cudaEventRecord(p3, stream1);

        // Layer 6: F1
        if (offset)
            fp_preact_f1_flag<<<int(4096 * offset), 1, 0, stream1>>>(d_flag_ac5, (float(*)[6][6])L_p3.output, L_f1.preact,
                                                                (float(*)[256][6][6])L_f1.weight, L_f1.bias);

        Aapply_step_function_flag<<<128, 128, 0, stream1>>>(d_flag_f1, L_f1.preact, L_f1.output, L_f1.O);
        if (offset != 1) cudaEventRecord(f1, stream1);

        // Layer 7: F2
        if (offset)
            fp_preact_f2_flag<<<int(4096 * offset), 1, 0, stream1>>>(d_flag_f1, L_f1.output, L_f2.preact, (float(*)[4096])L_f2.weight,
                                                                L_f2.bias);

        Aapply_step_function_flag<<<128, 128, 0, stream1>>>(d_flag_f2, L_f2.preact, L_f2.output, L_f2.O);
        if (offset != 1) cudaEventRecord(f2, stream1);

        // // Layer 8: F3
        if (offset)
            fp_preact_f3_flag<<<int(1000 * offset), 1, 0, stream1>>>(d_flag_f2, L_f2.output, L_f3.preact, (float(*)[4096])L_f3.weight,
                                                                L_f3.bias);

        Aapply_step_function_flag<<<128, 128, 0, stream1>>>(d_flag_f3, L_f3.preact, L_f3.output, L_f3.O);
    
    });

    // ----------------------------------------------------------------
    // CPU
    // ----------------------------------------------------------------
    {
        // Layer 1: C1 -> P1
        if (offset != 1) {
            fp_preact_c1_cpu((float(*)[227][3])L_input.output, (float(*)[55][55])L_c1.preact,
                         (float(*)[11][11][3])L_c1.weight, L_c1.bias);
        }
        *h_flag_ac1 = 1;

        // Layer 2: C2 -> P2
        if (offset != 1) {
            while (cudaEventQuery(p1) == cudaErrorNotReady);
            fp_preact_c2_cpu((float(*)[31][31])L_p1.output, (float(*)[27][27])L_c2.preact, (float(*)[96][5][5])L_c2.weight,
                         L_c2.bias);
        }
        *h_flag_ac2 = 1;

        // Layer 3: C3
        if (offset != 1) {
            while (cudaEventQuery(p2) == cudaErrorNotReady);
            fp_preact_c3_cpu((float(*)[15][15])L_p2.output, (float(*)[13][13])L_c3.preact, (float(*)[256][3][3])L_c3.weight,
                         L_c3.bias);
        }
        *h_flag_ac3 = 1;

        // Layer 4: C4
        if (offset != 1) {
            while (cudaEventQuery(c3) == cudaErrorNotReady);
            fp_preact_c4_cpu((float(*)[13][13])L_c3.output, (float(*)[13][13])L_c4.preact, (float(*)[384][3][3])L_c4.weight,
                         L_c4.bias);
        }
        *h_flag_ac4 = 1;

        // Layer 5: C5 -> P3
        if (offset != 1) {
            while (cudaEventQuery(c4) == cudaErrorNotReady);
            fp_preact_c5_cpu((float(*)[13][13])L_c4.output, (float(*)[13][13])L_c5.preact, (float(*)[384][3][3])L_c5.weight,
                         L_c5.bias);
        }
        *h_flag_ac5 = 1;

        // Layer 6: F1
        if (offset != 1) {
            while (cudaEventQuery(p3) == cudaErrorNotReady);
            fp_preact_f1_cpu((float(*)[6][6])L_p3.output, L_f1.preact, (float(*)[256][6][6])L_f1.weight, L_f1.bias);
        }
        *h_flag_f1 = 1;

        // Layer 7: F2
        if (offset != 1) {
            while (cudaEventQuery(f1) == cudaErrorNotReady);
            fp_preact_f2_cpu(L_f1.output, L_f2.preact, (float(*)[4096])L_f2.weight, L_f2.bias);
        }
        *h_flag_f2 = 1;

        // Layer 8: F3
        if (offset != 1) {
            while (cudaEventQuery(f2) == cudaErrorNotReady);
            fp_preact_f3_cpu(L_f2.output, L_f3.preact, (float(*)[4096])L_f3.weight, L_f3.bias);
        }
        *h_flag_f3 = 1;
    };

    gpu_thread.get();
    cudaDeviceSynchronize();
    
    double end = gettime();

    // Cleanup events
    if (offset != 1) {
        cudaEventDestroy(p1);
        cudaEventDestroy(p2);
        cudaEventDestroy(p3);
        cudaEventDestroy(f1);
        cudaEventDestroy(f2);
        cudaEventDestroy(c3);
        cudaEventDestroy(c4);
    }
    
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
