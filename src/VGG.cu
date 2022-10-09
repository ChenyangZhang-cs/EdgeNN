#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define Mask_width 3
#define Mask_height 3
#define Mask_radius_x Mask_width / 2
#define Mask_radius_y Mask_height / 2
#define TILE_WIDTH 32 // 16 X 16 TILE
#define B_x (TILE_WIDTH + Mask_width - 1)
#define B_y (TILE_WIDTH + Mask_height - 1)
#define clamp(x) (max(max((x), 0.0), x))
#define SIZE 224
#define max4(w, x, y, z) max(max(max(w, x), y), z)

#define INPUT_CHANNELS 3
#define CONV_SIZE 3
int layers[13][4] = {
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};

int dense[3][2] = {
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};

#define MASK_WIDTH 3
#define TILE_SIZE_1 32

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

typedef enum {
    CONV_1 = 512,
} ch;
const int out = ch(CONV_1);

float *dense_1;
float *dense_2;
float *dense_3;
float *bias_1;
float *bias_2;
float *bias_3;

void softmax(float *prediction, int classes) {
    int i;
    float max_val, sum;
    max_val = prediction[0];
    for (i = 1; i < classes; i++) {
        if (prediction[i] > max_val)
            max_val = prediction[i];
    }
    sum = 0.0;
    for (i = 0; i < classes; i++) {
        prediction[i] = exp(prediction[i] - max_val);
        sum += prediction[i];
    }
    for (i = 0; i < classes; i++) {
        prediction[i] /= sum;
    }
}

void normalizeBGR(float *hostInputImageData) {
    float coef[3] = {103.939, 116.779, 123.68};
    FILE *input = fopen("../data/VGG/vol.txt", "r");
    float dval, val;
    int count = 0;
    for (int i = 0; i < SIZE * SIZE * INPUT_CHANNELS; i++) {
        fscanf(input, "%f", &dval);
        // int n = (i+1);
        if (count == 0)
            val = dval - coef[count];
        if (count == 1)
            val = dval - coef[count];
        if (count == 2)
            val = dval - coef[count];
        hostInputImageData[i] = val;
        count++;
        if (count == 3)
            count = 0;
    }
    fclose(input);
}

void readWeights(int level, float *wconv, float *bias) {
    float dval;
    FILE *weight;
    weight = fopen("../data/VGG/vgg16_weights.txt", "r");
    if (weight == NULL) {
        exit(1);
    }
    if (level != 0) {
        for (int s = 0; s < level; s++) {
            for (int j = 1; j <= layers[s][0] * layers[s][1]; j++) {
                for (int k = 0; k < CONV_SIZE * CONV_SIZE; k++) {
                    fscanf(weight, "%f", &dval);
                }
            }
            for (int i = 0; i < layers[s][0]; i++) {
                fscanf(weight, "%f", &dval);
            }
        }
    }

    for (int j = 1; j <= layers[level][0] * layers[level][1]; j++) {
        for (int k = 0; k < CONV_SIZE * CONV_SIZE; k++) {
            fscanf(weight, "%f", &dval);
            *(wconv + j * CONV_SIZE * CONV_SIZE - 1 - k) = dval;
        }
    }

    int i = 0;
    while (i < layers[level][0]) {
        fscanf(weight, "%f", &dval);
        bias[i] = dval;
        i++;
    }

    if (level == 12) {
        int i, j;
        for (i = 0; i < dense[0][0]; i++) {
            for (j = 0; j < dense[0][1]; j++) {
                fscanf(weight, "%f", &dval);
                *(dense_1 + i) = dval;
            }
        }
        for (i = 0; i < dense[0][1]; i++) {
            fscanf(weight, "%f", &dval);
            *(bias_1 + i) = dval;
        }
        for (i = 0; i < dense[1][0]; i++) {
            for (j = 0; j < dense[1][1]; j++) {
                fscanf(weight, "%f", &dval);
                *(dense_2 + i) = dval;
            }
        }
        for (i = 0; i < dense[1][1]; i++) {
            fscanf(weight, "%f", &dval);
            *(bias_2 + i) = dval;
        }
        for (i = 0; i < dense[2][0]; i++) {
            for (j = 0; j < dense[2][1]; j++) {
                fscanf(weight, "%f", &dval);
                *(dense_3 + i) = dval;
            }
        }
        for (i = 0; i < dense[2][1]; i++) {
            fscanf(weight, "%f", &dval);
            *(bias_3 + i) = dval;
        }
    }
    fclose(weight);
}

// Maxpool layer definition
__global__ void maxpool(float *image, float *output, int number_of_channels, int image_height, int image_width,
                        int blockwidth) {
    __shared__ float Ns[32][32];
    for (int curr_channel = 0; curr_channel < number_of_channels; curr_channel++) {
        Ns[threadIdx.x][threadIdx.y] =
            image[(threadIdx.y * number_of_channels + curr_channel + blockIdx.y * (blockwidth * number_of_channels)) +
                  (threadIdx.x + blockIdx.x * blockwidth) * (image_width * number_of_channels)];
        __syncthreads();
        if ((threadIdx.x % 2 == 0) && (threadIdx.y % 2 == 0)) {
            output[blockIdx.y * (blockwidth / 2) * number_of_channels + (threadIdx.y / 2) * number_of_channels +
                   curr_channel +
                   (blockIdx.x * blockwidth / 2 + threadIdx.x / 2) * (image_width / 2) * number_of_channels] =
                max4(Ns[threadIdx.x][threadIdx.y], Ns[threadIdx.x][threadIdx.y + 1], Ns[threadIdx.x + 1][threadIdx.y],
                     Ns[threadIdx.x + 1][threadIdx.y + 1]);
        }
    }
}

void maxpool_CPU(float *image, float *output, int gpu_channels, int number_of_channels, int image_height,
                 int image_width) {
#pragma omp parallel for
    for (int k = gpu_channels; k < number_of_channels; k++) {
        for (int i = 0; i < image_height; i += 2) {
            for (int j = 0; j < image_width; j += 2) {
                int out_idx = k * image_width * image_height / 4 + i / 2 * image_width / 2 + j / 2;
                int idx = k * image_width * image_height;
                int idx1 = idx + i * image_width + j;
                int idx2 = idx + (i + 1) * image_width + j;
                int idx3 = idx + i * image_width + j + 1;
                int idx4 = idx + (i + 1) * image_width + j + 1;
                output[out_idx] = max4(image[idx1], image[idx2], image[idx3], image[idx4]);
            }
        }
    }
}

void convolution_CPU(float *I, const float *M, float *P, float *b, int channels, int width, int height,
                     int gpu_channels, int numberofOutputChannels) {
    int mask_size = Mask_width * Mask_width;
#pragma omp parallel for
    for (int c = gpu_channels; c < numberofOutputChannels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float tmp = 0;
                for (int ci = 0; ci < channels; ci++) {
                    for (int k = 0; k < mask_size; k++)
                        tmp +=
                            I[ci * height * width + i * width + j] * M[c * mask_size * channels + ci * mask_size + k];
                }
                P[c * height * width + i * width + j] = tmp;
            }
        }
    }
}

__global__ void fully1(float *I, const float *__restrict__ M, float *P, int channels, int numberofOutputChannels,
                       float *b) {
    __shared__ float F_ds[7][7];
    float acc[4096] = {0};
    int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;
    int threadNum = blockDim.x * blockDim.y;

    for (int current_channel = 0; current_channel < channels; current_channel++) {
        F_ds[threadIdx.x][threadIdx.y] = I[(threadIdx.x + (blockIdx.x * TILE_WIDTH)) * (7 * channels) +
                                           threadIdx.y * channels + current_channel + blockIdx.y * (TILE_WIDTH)]; //
        __syncthreads();
        for (int z = threadId; z < numberofOutputChannels; z += threadNum) {
            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 7; j++) {
                    acc[z] += F_ds[i][j] * M[z * 7 * 7 * 512 + current_channel * 7 * 7 + i * 7 + j];
                }
            }
        }
    }
    for (int z = threadId; z < numberofOutputChannels; z += threadNum) {
        P[z] = acc[z] + b[z];
    }
}

void fully1_CPU(float *I, const float *M, float *P, int channels, int gpu_channels, int numberofOutputChannels,
                float *b) {
    float acc[4096] = {0};
#pragma omp parallel for
    for (int current_channel = 0; current_channel < channels; current_channel++) {
        for (int z = gpu_channels; z < numberofOutputChannels; z++) {
            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 7; j++) {
                    acc[z] += I[i * (7 * channels) + j * channels + current_channel] *
                              M[z * 7 * 7 * 512 + current_channel * 7 * 7 + i * 7 + j];
                }
            }
        }
    }
#pragma omp parallel for
    for (int z = gpu_channels; z < numberofOutputChannels; z++) {
        P[z] = acc[z] + b[z];
    }
}

__global__ void fully2(float *I, const float *__restrict__ M, float *P, int channels, int gpu_output, float *b) {
    __shared__ float F_ds[4][32][32];
    int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;
    int threadNum = blockDim.x * blockDim.y;

    float acc[4096] = {0};
    for (int i = 0; i < 4; i++) {
        F_ds[i][threadIdx.x][threadIdx.y] = I[threadIdx.y + threadIdx.x * 32 + i * 32 * 32];
    }
    __syncthreads();

    int i, j, k;
    for (int current_op_channel = threadId; current_op_channel < gpu_output; current_op_channel += threadNum) {
        for (int current_channel = 0; current_channel < channels; current_channel++) {
            i = current_channel / 1024;
            j = ((current_channel) / 32) % 32;
            k = current_channel % 32;
            acc[current_op_channel] += F_ds[i][j][k] * M[current_channel + current_op_channel * channels];
        }
    }
    __syncthreads();
    for (int z = threadId; z < gpu_output; z += threadNum) {
        P[z] = clamp(acc[z] + b[z]);
    }
}

void fully2_CPU(float *I, const float *M, float *P, int channels, int gpu_output, int numberofOutputChannels,
                float *b) {
    float acc[4096] = {0};
    int i, j, k;
    for (int current_op_channel = gpu_output; current_op_channel < numberofOutputChannels; current_op_channel++) {
        for (int current_channel = 0; current_channel < channels; current_channel++) {
            i = current_channel / 1024;
            j = ((current_channel) / 32) % 32;
            k = current_channel % 32;
            acc[current_op_channel] += I[k + j * 32 + i * 32 * 32] * M[current_channel + current_op_channel * channels];
        }
    }
    for (int z = gpu_output; z < numberofOutputChannels; z++) {
        P[z] = clamp(acc[z] + b[z]);
    }
}

__global__ void fully3(float *I, const float *__restrict__ M, float *P, int channels, int numberofOutputChannels,
                       float *b) {
    __shared__ float F_ds[4][32][32];
    float acc[1000] = {0};
    for (int i = 0; i < 4; i++) {
        F_ds[i][threadIdx.x][threadIdx.y] = I[threadIdx.y + threadIdx.x * 32 + i * 32 * 32];
    }
    __syncthreads();
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        int i, j, k;
        for (int current_op_channel = 0; current_op_channel < numberofOutputChannels; current_op_channel++) {
            for (int current_channel = 0; current_channel < channels; current_channel++) {
                i = current_channel / 1024;
                j = ((current_channel) / 32) % 32;
                k = current_channel % 32;
                acc[current_op_channel] += F_ds[i][j][k] * M[current_channel + current_op_channel * channels];
            }
        }
        for (int z = 0; z < numberofOutputChannels; z++) {
            P[z] = clamp(acc[z] + b[z]);
        }
    }
}

__global__ void convolution(float *I, const float *__restrict__ M, float *P, float *b, int channels, int width,
                            int height, int numberofOutputChannels) {
    __shared__ float N_ds[B_y][B_x];
    int dest_Y;
    int dest_X;
    int src_X;
    int src_Y;
    int src;
    int dest;

    float accum[out] = {0};

    for (int current_channel = 0; current_channel < channels; current_channel++) {
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x, dest_Y = dest / B_x, dest_X = dest % B_x,
        src_Y = blockIdx.y * TILE_WIDTH + dest_Y - Mask_radius_x,
        src_X = blockIdx.x * TILE_WIDTH + dest_X - Mask_radius_y,
        src = (src_Y * width + src_X) * channels + current_channel;
        if (src_Y >= 0 && src_Y < height && src_X >= 0 && src_X < width)
            N_ds[dest_Y][dest_X] = I[src];
        else
            N_ds[dest_Y][dest_X] = 0.0;

        for (int iter = 1; iter <= (B_x * B_y) / (TILE_WIDTH * TILE_WIDTH); iter++) {
            // Second batch loading
            dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter * (TILE_WIDTH * TILE_WIDTH);
            dest_Y = dest / B_x, dest_X = dest % B_x;
            src_Y = blockIdx.y * TILE_WIDTH + dest_Y - Mask_radius_x;
            src_X = blockIdx.x * TILE_WIDTH + dest_X - Mask_radius_y;
            src = (src_Y * width + src_X) * channels + current_channel;
            if (dest_Y < B_y && dest_X < B_x) {
                if (src_Y >= 0 && src_Y < height && src_X >= 0 && src_X < width)
                    N_ds[dest_Y][dest_X] = I[src];
                else
                    N_ds[dest_Y][dest_X] = 0.0;
            }
        }
        __syncthreads();

        int y, x, z;
        for (z = 0; z < numberofOutputChannels; z++)
            for (y = 0; y < Mask_width; y++)
                for (x = 0; x < Mask_width; x++)
                    accum[z] +=
                        N_ds[threadIdx.y + y][threadIdx.x + x] *
                        M[(z * Mask_width * Mask_width * channels + current_channel * Mask_width * Mask_width) +
                          y * Mask_width + x];


        __syncthreads();
    }

    int y, x, z;
    y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width)
        // add bias and relu
        for (z = 0; z < numberofOutputChannels; z++)
            P[(y * width * numberofOutputChannels + numberofOutputChannels * x) + z] = clamp(accum[z] + b[z]);
}

int main(int argc, char **argv) {
    int cpu_offset = 0;
    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            if (argv[i][0] == '-') {
                switch (argv[i][1]) {
                case 'o':
                    i++;
                    cpu_offset = atoi(argv[i]);
                    break;
                }
            }
        }
    }
    double t1, t2;
    double read_w_time = 0;
    cudaError_t err = cudaSuccess;

    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        printf("failed to reset device \n");
        exit(1);
    }

    float time_taken = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int mask_Rows = Mask_height; // Set it as per requirement of 64 X 32
    int mask_cols = Mask_width;

    int numberofImageChannels = 3;
    int numberofOutputChannels = 64;
    int width_image = SIZE;
    int height_image = SIZE;

    float *host_Image_output;
    float *deviceOutputImageData_1_1;
    float *device_maxpool_output;
    float *bias;
    float *biasDense = (float *)malloc(sizeof(float) * dense[0][1]);

    double startTime = gettime();

    err = cudaMallocManaged((void **)&bias, layers[12][0] * sizeof(float), cudaMemAttachHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*************************** conv1-1 ******************************/

    int level = 0;
    // layer parameters
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *hostMaskData;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;
    err = cudaMalloc((void **)&deviceOutputImageData_1_1,
                     width_image * height_image * numberofOutputChannels * sizeof(float));

    float *hostInputImageData;
    err = cudaMallocManaged((void **)&hostInputImageData,
                            width_image * height_image * numberofImageChannels * sizeof(float));
    t1 = gettime();
    normalizeBGR(hostInputImageData);
    t2 = gettime();
    read_w_time += t2 - t1;
    dim3 dimGrid(((width_image - 1) / TILE_WIDTH) + 1, ((height_image - 1) / TILE_WIDTH) + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    cudaEventRecord(start);
    convolution<<<dimGrid, dimBlock>>>(hostInputImageData, hostMaskData, deviceOutputImageData_1_1, bias,
                                       numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    cudaFree(hostInputImageData);
    cudaFree(hostMaskData);

    /*************************** conv1-1 end******************************/
    /*************************** conv1-2 start ******************************/
    level = 1;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    err = cudaMallocManaged((void **)&hostMaskData,
                            numberofOutputChannels * numberofImageChannels * CONV_SIZE * CONV_SIZE * sizeof(float));
    float *deviceOutputImageData_1_2;
    err = cudaMalloc((void **)&deviceOutputImageData_1_2,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;
    convolution<<<dimGrid, dimBlock>>>(deviceOutputImageData_1_1, hostMaskData, deviceOutputImageData_1_2, bias,
                                       numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    float *hostOutputImageData_1_2 =
        (float *)malloc(width_image * height_image * numberofOutputChannels * sizeof(float));

    cudaMemcpy(hostOutputImageData_1_2, deviceOutputImageData_1_2,
               width_image * height_image * numberofOutputChannels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceOutputImageData_1_1);
    cudaFree(hostMaskData);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    /*************************** conv1-2 end ******************************/
    /*************************** conv1-maxpool start ******************************/

    err = cudaMalloc((void **)&device_maxpool_output,
                     width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device_image_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int blockwidth = 32;
    int number_blocks = width_image / blockwidth;
    dim3 dimGrid_m1(number_blocks, number_blocks, 1);
    dim3 dimBlock_m1(blockwidth, blockwidth, 1);
    float *host_maxpool_output =
        (float *)malloc(width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float));

    maxpool<<<dimGrid_m1, dimBlock_m1>>>(deviceOutputImageData_1_2, device_maxpool_output, numberofOutputChannels,
                                         height_image, width_image, blockwidth);
    cudaDeviceSynchronize();
    free(hostOutputImageData_1_2);
    free(host_maxpool_output);
    cudaFree(deviceOutputImageData_1_2);
    /*************************** conv1-maxpool end ******************************/
    /*******************************conv_2_1 start**********************************************/
    width_image /= 2;
    height_image /= 2;
    level = 2;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *deviceOutputImageData_2_1;
    err = cudaMalloc((void **)&deviceOutputImageData_2_1,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    dim3 dimGrid_2(((width_image - 1) / TILE_WIDTH) + 1, ((height_image - 1) / TILE_WIDTH) + 1, 1);
    dim3 dimBlock_2(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid_2, dimBlock_2>>>(device_maxpool_output, hostMaskData, deviceOutputImageData_2_1, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(device_maxpool_output);

    /*******************************conv_2_1 end**********************************************/

    /******************************conv_2_2 start********************************************/

    level = 3;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *deviceOutputImageData_2_2;
    err = cudaMalloc((void **)&deviceOutputImageData_2_2,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    convolution<<<dimGrid_2, dimBlock_2>>>(deviceOutputImageData_2_1, hostMaskData, deviceOutputImageData_2_2, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);

    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputImageData_2_1);

    /******************************conv_2_2 end*********************************************/

    /******************************max2 start**********************************************/
    float *deviceOutputMaxPooledData2;
    err = cudaMalloc((void **)&deviceOutputMaxPooledData2,
                     width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device_image_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    blockwidth = 16;
    number_blocks = width_image / blockwidth;
    dim3 dimGrid_m2(number_blocks, number_blocks, 1);
    dim3 dimBlock_m2(blockwidth, blockwidth, 1);

    maxpool<<<dimGrid_m2, dimBlock_m2>>>(deviceOutputImageData_2_2, deviceOutputMaxPooledData2, numberofOutputChannels,
                                         height_image, width_image, blockwidth);

    cudaDeviceSynchronize();
    cudaFree(deviceOutputImageData_2_2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    /*****************************max2 end*************************************************/

    /*****************************conv_3_1 start************************************************/
    width_image /= 2;
    height_image /= 2;
    level = 4;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_3_1;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData_3_1,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    dim3 dimGrid_3(((width_image - 1) / TILE_WIDTH) + 1, ((height_image - 1) / TILE_WIDTH) + 1, 1);
    dim3 dimBlock_3(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid_3, dimBlock_3>>>(deviceOutputMaxPooledData2, hostMaskData, deviceOutputImageData_3_1, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputMaxPooledData2);
    // /*****************************conv_3_1 end************************************************/
    //
    // /****************************conv_3_2 start**********************************************/
    level = 5;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_3_2;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData_3_2,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    convolution<<<dimGrid_3, dimBlock_3>>>(deviceOutputImageData_3_1, hostMaskData, deviceOutputImageData_3_2, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputImageData_3_1);
    /***************************conv_3_2 end************************************************/
    /***************************conv_3_3 start************************************************/
    level = 6;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_3_3;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&deviceOutputImageData_3_3,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    convolution<<<dimGrid_3, dimBlock_3>>>(deviceOutputImageData_3_2, hostMaskData, deviceOutputImageData_3_3, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);

    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputImageData_3_2);
    /***************************conv_3_3 end************************************************/

    /******************************max3 start**********************************************/
    float *deviceOutputMaxPooledData3;
    err = cudaMalloc((void **)&deviceOutputMaxPooledData3,
                     width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device_image_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    blockwidth = 8;
    number_blocks = width_image / blockwidth;
    dim3 dimGrid_m3(number_blocks, number_blocks, 1);
    dim3 dimBlock_m3(blockwidth, blockwidth, 1);
    maxpool<<<dimGrid_m3, dimBlock_m3>>>(deviceOutputImageData_3_3, deviceOutputMaxPooledData3, numberofOutputChannels,
                                         height_image, width_image, blockwidth);
    cudaDeviceSynchronize();
    cudaFree(deviceOutputImageData_3_3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    /*****************************max3 end*************************************************/

    /*****************************conv_4_1 start************************************************/
    width_image /= 2;
    height_image /= 2;
    level = 7;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_4_1;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData_4_1,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    dim3 dimGrid_4(((width_image - 1) / TILE_WIDTH) + 1, ((height_image - 1) / TILE_WIDTH) + 1, 1);
    dim3 dimBlock_4(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid_4, dimBlock_4>>>(deviceOutputMaxPooledData3, hostMaskData, deviceOutputImageData_4_1, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);

    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputMaxPooledData3);
    /*****************************conv_4_1 end************************************************/

    /****************************conv_4_2 start**********************************************/
    level = 8;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_4_2;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData_4_2,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    convolution<<<dimGrid_4, dimBlock_4>>>(deviceOutputImageData_4_1, hostMaskData, deviceOutputImageData_4_2, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputImageData_4_1);
    /***************************conv_4_2 end************************************************/

    /***************************conv_4_3 start************************************************/
    level = 9;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_4_3;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData_4_3,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    convolution<<<dimGrid_4, dimBlock_4>>>(deviceOutputImageData_4_2, hostMaskData, deviceOutputImageData_4_3, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputImageData_4_2);
    /***************************conv_4_3 end************************************************/

    /******************************max4 start**********************************************/
    float *deviceOutputMaxPooledData4;
    err = cudaMalloc((void **)&deviceOutputMaxPooledData4,
                     width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputMaxPooledData4 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // image 28
    blockwidth = 4;
    number_blocks = width_image / blockwidth;
    dim3 dimGrid_m4(number_blocks, number_blocks, 1);
    dim3 dimBlock_m4(blockwidth, blockwidth, 1);
    maxpool<<<dimGrid_m4, dimBlock_m4>>>(deviceOutputImageData_4_3, deviceOutputMaxPooledData4, numberofOutputChannels,
                                         height_image, width_image, blockwidth);
    cudaDeviceSynchronize();
    cudaFree(deviceOutputImageData_4_3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);

    /*****************************max4 end*************************************************/
    /*****************************conv_5_1 start************************************************/
    width_image /= 2;
    height_image /= 2;
    level = 10;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_5_1;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&deviceOutputImageData_5_1,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    dim3 dimGrid_5(((width_image - 1) / TILE_WIDTH) + 1, ((height_image - 1) / TILE_WIDTH) + 1, 1);
    dim3 dimBlock_5(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid_5, dimBlock_5>>>(deviceOutputMaxPooledData4, hostMaskData, deviceOutputImageData_5_1, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy 5.1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputMaxPooledData4);
    /*****************************conv_5_1 end************************************************/
    /****************************conv_5_2 start**********************************************/
    level = 11;
    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_5_2;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged((void **)&deviceOutputImageData_5_2,
                            width_image * height_image * numberofOutputChannels * sizeof(float), cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;
    convolution<<<dimGrid_5, dimBlock_5>>>(deviceOutputImageData_5_1, hostMaskData, deviceOutputImageData_5_2, bias,
                                           numberofImageChannels, width_image, height_image, numberofOutputChannels);
    cudaDeviceSynchronize();
    cudaFree(hostMaskData);
    cudaFree(deviceOutputImageData_5_1);
    /***************************conv_5_2 end************************************************/

    /***************************conv_5_3 start************************************************/
    level = 12;
    err = cudaMallocManaged((void **)&bias_1, dense[0][1] * sizeof(float), cudaMemAttachHost);
    err = cudaMallocManaged((void **)&bias_2, dense[1][1] * sizeof(float), cudaMemAttachHost);
    err = cudaMallocManaged((void **)&bias_3, dense[2][1] * sizeof(float));
    err = cudaMallocManaged((void **)&dense_1, dense[0][0] * dense[0][1] * sizeof(float), cudaMemAttachHost);
    err = cudaMallocManaged((void **)&dense_2, dense[1][0] * dense[1][1] * sizeof(float), cudaMemAttachHost);
    err = cudaMallocManaged((void **)&dense_3, dense[2][0] * dense[2][1] * sizeof(float));

    numberofOutputChannels = layers[level][0];
    numberofImageChannels = layers[level][1];
    float *deviceOutputImageData_5_3;
    err = cudaMallocManaged((void **)&hostMaskData,
                            mask_Rows * mask_cols * numberofImageChannels * numberofOutputChannels * sizeof(float),
                            cudaMemAttachHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate hostMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData_5_3,
                     width_image * height_image * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t1 = gettime();
    readWeights(level, hostMaskData, bias);
    t2 = gettime();
    read_w_time += t2 - t1;

    int gpu_output = numberofOutputChannels * (100 - cpu_offset) / 100;
    float *hostOutputImageData_5_3 =
        (float *)malloc(width_image * height_image * numberofOutputChannels * sizeof(float));
    float *hostOutputImageData_5_2 =
        (float *)malloc(width_image * height_image * numberofOutputChannels * sizeof(float));

    if (gpu_output > 0)
        convolution<<<dimGrid_5, dimBlock_5>>>(deviceOutputImageData_5_2, hostMaskData, deviceOutputImageData_5_3, bias,
                                               numberofImageChannels, width_image, height_image, gpu_output);
    if (cpu_offset > 0)
        convolution_CPU(deviceOutputImageData_5_2, hostMaskData, hostOutputImageData_5_3, bias, numberofImageChannels,
                        width_image, height_image, gpu_output, numberofOutputChannels);
    cudaDeviceSynchronize();
    if (cpu_offset > 0) {
        cudaMemcpy(deviceOutputImageData_5_3 + width_image * height_image * gpu_output, hostOutputImageData_5_3,
                   width_image * height_image * (numberofOutputChannels - gpu_output) * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    cudaFree(hostMaskData);
    cudaFree(deviceOutputImageData_5_2);
    /***************************conv_5_3 end************************************************/
    /******************************max5 start**********************************************/
    float *deviceOutputMaxPooledData5;
    err = cudaMalloc((void **)&deviceOutputMaxPooledData5,
                     width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputMaxPooledData5 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    blockwidth = 2;
    number_blocks = width_image / blockwidth;
    dim3 dimGrid_m5(number_blocks, number_blocks, 1);
    dim3 dimBlock_m5(blockwidth, blockwidth, 1);
    maxpool<<<dimGrid_m5, dimBlock_m5>>>(deviceOutputImageData_5_3, deviceOutputMaxPooledData5, numberofOutputChannels,
                                         height_image, width_image, blockwidth);
    cudaDeviceSynchronize();
    float *hostOutputMaxPooledData5 =
        (float *)malloc(width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float));
    cudaMemcpy(hostOutputMaxPooledData5, deviceOutputMaxPooledData5,
               width_image / 2 * height_image / 2 * numberofOutputChannels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceOutputImageData_5_3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);

    /*****************************max5 end*************************************************/
    /*****************************dense_1_1 start************************************************/
    width_image /= 2;
    height_image /= 2;
    level = 0;
    int output = dense[level][1];
    float *deviceOutputImageDataDense_1_1;
    err = cudaMalloc((void **)&deviceOutputImageDataDense_1_1, output * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *hostOutputImageDataDense_1_1 = (float *)malloc(output * sizeof(float));
    dim3 dimGrid_fc1_1(1, 1, 1);
    dim3 dimBlock_fc1_1(width_image, width_image, 1);
    gpu_output = output * (100 - cpu_offset) / 100;
    t1 = gettime();
    if (gpu_output > 0)
        fully1<<<dimGrid_fc1_1, dimBlock_fc1_1>>>(deviceOutputMaxPooledData5, dense_1, deviceOutputImageDataDense_1_1,
                                                  512, gpu_output, bias_1);
    if (cpu_offset > 0)
        fully1_CPU(hostOutputMaxPooledData5, dense_1, hostOutputImageDataDense_1_1, 512, gpu_output, output, bias_1);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceOutputImageDataDense_1_1 + gpu_output, hostOutputImageDataDense_1_1,
               (output - gpu_output) * sizeof(float), cudaMemcpyHostToDevice);
    t2 = gettime();
    cudaMemcpy(hostOutputImageDataDense_1_1, deviceOutputImageDataDense_1_1, gpu_output * sizeof(float),
               cudaMemcpyDeviceToHost);
    free(hostOutputMaxPooledData5);

    cudaDeviceSynchronize();

    cudaFree(deviceOutputMaxPooledData5);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);

    /*****************************dense_1_1 end************************************************/
    /*****************************dense_1_2 start************************************************/
    level = 1;
    output = dense[level][1];
    float *deviceOutputImageDataDense_1_2;
    err = cudaMalloc((void **)&deviceOutputImageDataDense_1_2, output * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *hostOutputImageDataDense_1_2 = (float *)malloc(output * sizeof(float));

    dim3 dimGrid_fc1_2(1, 1, 1);
    dim3 dimBlock_fc1_2(32, 32, 1);
    gpu_output = output * (100 - cpu_offset) / 100;
    fully2<<<dimGrid_fc1_2, dimBlock_fc1_2>>>(deviceOutputImageDataDense_1_1, dense_2, deviceOutputImageDataDense_1_2,
                                              4096, output, bias_2);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy FC 1.2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    free(hostOutputImageDataDense_1_1);
    free(hostOutputImageDataDense_1_2);
    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy FC 1.2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    cudaFree(deviceOutputImageDataDense_1_1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    /*****************************dense_1_2 end************************************************/

    /*****************************dense_1_3 start************************************************/

    level = 2;
    output = dense[level][1];

    err = cudaMallocManaged((void **)&host_Image_output, output * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host_Image_output(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 dimGrid_fc1_3(1, 1, 1);
    dim3 dimBlock_fc1_3(32, 32, 1);
    fully3<<<dimGrid_fc1_3, dimBlock_fc1_3>>>(deviceOutputImageDataDense_1_2, dense_3, host_Image_output, 4096, output,
                                              bias_3);
    cudaDeviceSynchronize();
    softmax(host_Image_output, 1000);

    cudaFree(host_Image_output);
    cudaFree(deviceOutputImageDataDense_1_2);
    cudaFree(dense_1);
    cudaFree(dense_2);
    cudaFree(dense_3);
    cudaFree(bias);
    cudaFree(bias_1);
    cudaFree(bias_2);
    cudaFree(bias_3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    double endTime = gettime();
    printf("%d\t%lf\n", cpu_offset, (endTime - startTime - read_w_time));

    return 0;
}
