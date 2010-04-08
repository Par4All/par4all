CUC = nvcc

all: $(EXECUTABLE)

# This should be more configurable:
NVIDIA_SDK_DIR=$(HOME)/NVIDIA_GPU_Computing_SDK
CUDA_DIR=/usr/local/cuda

# Add runtime:
OBJFILES += p4a_accel.o

p4a_accel.cu: $(P4A_ACCEL_DIR)/p4a_accel.cu
	ln -s $<

CPPFLAGS = -I.. -I.  -I../../../../P4A_CUDA -DP4A_ACCEL_CUDA -DP4A_DEBUG -I$(CUDA_DIR)/include -I$(NVIDIA_SDK_DIR)/C/common/inc -DUNIX
CUFLAGS += --compiler-options -fno-strict-aliasing --ptxas-options=-v -arch=sm_13 -O2 -c

LDFLAGS = -fPIC -L$(CUDA_DIR)/lib64 -L$(NVIDIA_SDK_DIR)/C/lib -L$(NVIDIA_SDK_DIR)/C/common/lib/linux

LDLIBS = -lcudart -lcutil

# New default rule to compile CUDA source files:
%.o: %.cu
	$(CUC) -c $(CPPFLAGS) $(CUFLAGS) $<

$(EXECUTABLE): $(OBJFILES)

clean::
	rm -f $(EXECUTABLE) $(OBJFILES)
