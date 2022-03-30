all:
	nvcc -Xcompiler -fopenmp -arch=compute_72 -code=sm_72 src/main.cu -o SqueezeNet

clean:
	rm SqueezeNet