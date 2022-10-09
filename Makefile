RM = rm
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fopenmp
BINDIR = bin
SRCDIR = src
OBJECTS = ${BINDIR}/AlexNet ${BINDIR}/FCNN ${BINDIR}/LeNet ${BINDIR}/ResNet ${BINDIR}/SqueezeNet ${BINDIR}/VGG
ALEXNETMAIN = AlexNet.cu
FCNNMAIN = FCNN.cu
LENETMAIN = LeNet.cu
RESNETMAIN = ResNet.cu
SQUEEZENETMAIN = SqueezeNet.cu
VGGMAIN = VGG.cu

all: ${OBJECTS}

${BINDIR}/AlexNet: ${SRCDIR}/${ALEXNETMAIN}
	@mkdir -p ${BINDIR}
	${NVCC} ${NVCCFLAGS} -lcuda -lcublas $^ -o $@

${BINDIR}/FCNN: ${SRCDIR}/${FCNNMAIN}
	@mkdir -p ${BINDIR}
	${NVCC} ${NVCCFLAGS} -Wno-deprecated-gpu-targets -lcuda -lcublas -arch=compute_72 -code=sm_72 $^ -o $@

${BINDIR}/LeNet: ${SRCDIR}/${LENETMAIN}
	@mkdir -p ${BINDIR}
	${NVCC} ${NVCCFLAGS} -Wno-deprecated-gpu-targets -lcuda -lcublas $^ -o $@

${BINDIR}/ResNet: ${SRCDIR}/${RESNETMAIN}
	@mkdir -p ${BINDIR}
	${NVCC} ${NVCCFLAGS} -lcuda -lcublas $^ -o $@ 

${BINDIR}/SqueezeNet: ${SRCDIR}/${SQUEEZENETMAIN}
	@mkdir -p ${BINDIR}
	${NVCC} ${NVCCFLAGS} -arch=compute_72 -code=sm_72 $^ -o $@

${BINDIR}/VGG: ${SRCDIR}/${VGGMAIN}
	${NVCC} ${NVCCFLAGS} $^ -o $@

clean:
	${RM} -f ${OBJECTS}