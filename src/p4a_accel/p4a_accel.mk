CC = gcc
CUC = nvcc
LINK       := gcc -fPIC
CUDA_DIR=/usr/local/cuda

EXECUTABLE =     $(TARGET:=-seq) 
EXECUTABLE-OMP = $(TARGET:=-omp) 
EXECUTABLE-CU  = $(TARGET:=-cuda) 
EXECUTABLE-CL  = $(TARGET:=-cl) 

EXECUTABLES = $(EXECUTABLE) $(EXECUTABLE-OMP) $(EXECUTABLE-CU) $(EXECUTABLE-CL) 

all: $(EXECUTABLES) 

seq : $(EXECUTABLE)
openmp : $(EXECUTABLE-OMP)
cuda : $(EXECUTABLE-CU)
opencl : $(EXECUTABLE-CL)

#Kernels are not included in C files in OpenCL ...
CFILES += p4a_accel.c

CUFILES = $(CFILES:.c=.cu) $(KERNELFILES:.c=.cu) 

CLFILES = $(KERNELFILES:.c=.cl)

# Add runtime:
OBJCUFILES = $(CUFILES:.cu=.cu.o)

p4a_accel.c: $(P4A_ACCEL_DIR)/p4a_accel.c
	ln -s $<

%.cl:%.c
	cpp -DP4A_ACCEL_OPENCL -I$(P4A_ACCEL_DIR) -P -fdirectives-only -nostdinc $< -o $@

%.cu:%.c
	ln -s $< $@ 

BASEFLAGS = -I$(P4A_ACCEL_DIR) -I.. -I.  -DP4A_PROFILING  -DUNIX 
#Flags for openMP mode
CFLAGS = $(BASEFLAGS) -DP4A_ACCEL_OPENMP -std=c99

#Flags for OpenCL mode 
CLFLAGS = $(BASEFLAGS) -DP4A_ACCEL_OPENCL -I$(CUDA_DIR)/include/CL -I$(CUDA_DIR)/include -std=c99 
LDFLAGS = -fPIC -L/usr/lib 
CLLIBS =  -lOpenCL 

#Flags for Cuda mode (nvcc ~ g++)
CPPFLAGS = $(BASEFLAGS) -DP4A_ACCEL_CUDA -I../../../../P4A_CUDA -I$(CUDA_DIR)/include 
CUFLAGS += --compiler-options -fno-strict-aliasing -O2 


# New default rule to compile CUDA source files:

$(EXECUTABLE): $(CFILES) $(KERNELFILES)
	$(CC) $(CFLAGS) -o $@ $(CFILES)  $(KERNELFILES)  

$(EXECUTABLE-OMP): $(CFILES) $(KERNELFILES)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(CFILES)  $(KERNELFILES) -fopenmp

# New default rule to compile CUDA source files:
%.cu.o: %.cu
	$(CUC) $(CPPFLAGS) $(CUFLAGS) -o $@ -c $< 

$(EXECUTABLE-CU): $(OBJCUFILES)
	$(CUC) -o $(EXECUTABLE-CU) $(OBJCUFILES) $(LDLIBS) 

# rule to compile OpenCL source files:
$(EXECUTABLE-CL): $(CFILES) $(CLFILES)
	$(CC) $(CLFLAGS) -o $@ $(CFILES) $(CLLIBS) $(LDFLAGS)

clean::
	rm -f $(EXECUTABLES) *.o $(CUFILES) $(CLFILES) *~ ./p4a_accel.c
