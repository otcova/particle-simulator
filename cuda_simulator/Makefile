IO_LIB = -I../particle_io/c_api/include/ -lparticle_io_c
DEBUG_FLAGS = -O3 -flto $(IO_LIB) -L../target/debug 
FLAGS = -O3 -flto $(IO_LIB) -L../target/release

SRC_FILES = $(shell find src -type f)

.PHONY: build
build: build/particle_simulator

.PHONY: debug
debug: build/particle_simulator_debug

.PHONY: clean
clean:
	rm -r build

build/particle_simulator: $(SRC_FILES) Makefile
	@mkdir -p build
	gcc src/cuda_simulator.c $(FLAGS) -o build/cuda_simulator 

build/particle_simulator_debug: $(SRC_FILES) Makefile
	@mkdir -p build
	gcc src/cuda_simulator.c $(DEBUG_FLAGS) -o build/cuda_simulator_debug 
