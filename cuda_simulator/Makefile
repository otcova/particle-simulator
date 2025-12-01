CUDA_HOME := $(or $(CUDA_HOME),$(wildcard /Soft/cuda/12.2.2),$(wildcard /opt/cuda),$(wildcard /usr/local/cuda))

# If still empty → throw an error
ifeq ($(CUDA_HOME),)
$(error Could not find a valid CUDA installation. You can set a custom path with CUDA_HOME environment variable)
endif

IO_LIB = -I../particle_io/c_api/include/ -lparticle_io_c
DEBUG_FLAGS = -O3 $(IO_LIB) -L../target/debug #-flto
FLAGS       = -O3 $(IO_LIB) -L../target/release #-flto

NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH        = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc --ptxas-options=-v 
LD_FLAGS    = -lcudart -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib  -Xlinker -rpath,$(CUDA_HOME)/lib64

SRC_FILES = $(shell find src -type f)

.PHONY: build
build: build/cuda_simulator

.PHONY: debug
debug: build/cuda_simulator_debug

.PHONY: clean
clean:
	rm -r build

build/cuda_simulator: $(SRC_FILES)
	@mkdir -p build
	$(NVCC) src/cuda_simulator.cu $(FLAGS) $(NVCC_FLAGS) $(LD_FLAGS) -o build/cuda_simulator

build/cuda_simulator_debug: $(SRC_FILES)
	@mkdir -p build
	$(NVCC) src/cuda_simulator.cu $(DEBUG_FLAGS) $(NVCC_FLAGS) $(LD_FLAGS) -o build/cuda_simulator_debug
