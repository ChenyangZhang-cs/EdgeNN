# EdgeNN

## Introduction

This repository provides the source code of manuscript *EdgeNN: Efficient Neural Network Inference for CPU-GPU Integrated Edge Devices*.

Before running VGG, you need to download the VGG weight file from https://mega.nz/file/LIhjXRhQ#scgNodAkfwWIUZdTcRfmKNHjtUfUb2KiIvfvXdIe-vc, decompress it, and put it into data/VGG.

## Abstract

With the development of the architectures and the growth of AIoT application requirements, data processing on edge becomes popular. Neural network inference is widely employed for data analytics on edge devices. This paper extensively explores neural network inference on integrated edge devices and proposes EdgeNN, the first neural network inference solution on CPU-GPU integrated edge devices. EdgeNN has three novel characteristics. First, EdgeNN can adaptively utilize the unified physical memory and conduct the zero-copy optimization. Second, EdgeNN involves a novel inference-targeted inter- and intrakernel CPU-GPU hybrid execution approach, which co-runs the CPU with the GPU to fully utilize the edge device’s computing resources. Third, EdgeNN adopts a fine-grained inference task distribution strategy, which can divide the complicated inference structure into sub-tasks mapped to the CPU and the GPU M.O2 adaptively. Experiments show that on six popular neural network inference tasks, EdgeNN brings an average of 3.97×, 3.12× and 8.80× speedups to inference on the CPU of the integrated device, inference on a mobile phone CPU, and inference on an edge CPU device. Additionally, it achieves 22.02% time benefits to the direct execution of the original programs. Specifically, 9.93% comes from better utilization of unified memory, and 10.76% comes from the task distribution between the CPU and the GPU. Besides, EdgeNN can deliver 29.14× and 5.70× higher energy efficiency than the edge CPU and the discrete GPU respectively. We have made EdgeNN available at https://github.com/ChenyangZhang-cs/EdgeNN.

## Build

1. Set up CUDA environment.

2. Complie all example program

```makefile
make
```

## Run

1. ``cd example``

2. Run all example programs:

```shell
bash run_all.sh
```

Or run one example program:

```shell
bash run_AlexNet.sh
bash run_FCNN.sh
bash run_LeNet.sh
bash run_ResNet.sh
bash run_SqueezeNet.sh
bash run_VGG.sh
```

To run all programs with maximum performance, the hardware should support cuda unified memory.

<!-- ## Acknowledgement -->

<!-- ## Citation -->
