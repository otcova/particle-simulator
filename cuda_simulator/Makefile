IO_LIB = -I../particle_io/c_api/include/ -lparticle_io_c
DEBUG_FLAGS = -O3 $(IO_LIB) -L../target/debug #-flto
FLAGS = -O3 $(IO_LIB) -L../target/release #-flto

CUDA_HOME   = /opt/cuda
NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH       = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

SRC_FILES = $(shell find src -type f)

.PHONY: build
build: build/particle_simulator

.PHONY: debug
debug: build/particle_simulator_debug

.PHONY: clean
clean:
	rm -r build

build/particle_simulator: $(SRC_FILES)
	@mkdir -p build
	$(NVCC) src/cuda_simulator.cu $(FLAGS) $(NVCC_FLAGS) $(LD_FLAGS) -o build/cuda_simulator

build/particle_simulator_debug: $(SRC_FILES)
	@mkdir -p build
	$(NVCC) src/cuda_simulator.cu $(DEBUG_FLAGS) $(NVCC_FLAGS) $(LD_FLAGS) -o build/cuda_simulator_debug
