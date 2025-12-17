# Compiler commands
NVIDIA_COMPILER = nvcc

# Common flags
COMMON_FLAGS = -O2 -std=c++17

DEBUG_FLAGS = -O0 -g -G -std=c++17
INCLUDE_DIRS = -Iinclude -Ilib -I/home/chon0705/boost_1_66_0/include
LIB_DIRS = -L/home/chon0705/boost_1_66_0/lib -L/usr/lib/x86_64-linux-gnu
LIBS = -lboost_program_options -lboost_filesystem -lboost_system -lstdc++

# CUDA-specific flags
CUDA_FLAGS = --disable-warnings -std=c++17 -Xcompiler "-O2 -fopenmp" --compiler-options -fopenmp -lcudart

CUDA_DEBUG_FLAGS = --disable-warnings -g -G -std=c++17 -Xcompiler "-O0 -fopenmp" --compiler-options -fopenmp -lcudart

# Source files
SOURCE_FILES := $(wildcard *.cu) $(wildcard source/gsptucker/*.cpp) $(wildcard source/common/*.cpp) 

# Output executables
CUDA_OUTPUT = GSPTucker

all: cuda_build

cuda_build: $(SOURCE_FILES)
		$(NVIDIA_COMPILER) $(CUDA_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(SOURCE_FILES) -o $(CUDA_OUTPUT)

# Debug build
debug_build: $(SOURCE_FILES)
		$(NVIDIA_COMPILER) $(CUDA_DEBUG_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(SOURCE_FILES) -o $(CUDA_OUTPUT)

.PHONY: clean
clean:
	rm -f $(CUDA_OUTPUT)
