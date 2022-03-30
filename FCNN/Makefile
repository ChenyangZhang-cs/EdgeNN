all:
	nvcc  -lcuda -lcublas -arch=compute_72 -code=sm_72 ./src/main.cu -o FCNN -Xcompiler -fopenmp -Wno-deprecated-gpu-targets
clean:
	rm FCNN
