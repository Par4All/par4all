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

#Creation of the kernel wrapper with header for OpenMP
WRAPFILES = $(KERNELFILES:.c=_wrapper.c)

#Kernels are not included in C files in OpenCL ...
CFILES += p4a_accel.c

CUFILES = $(CFILES:.c=.cu) $(WRAPFILES:.c=.cu) 

CLFILES = $(KERNELFILES:.c=_wrapper.cl)

# Add runtime:
OBJCUFILES = $(CUFILES:.cu=.cu.o)

p4a_accel.c: $(P4A_ACCEL_DIR)/p4a_accel.c
	ln -s $<

%_wrapper.c:%.c
	cat $(P4A_ACCEL_DIR)/p4a_accel_wrapper-OpenMP.h $< > $@

#Specification of _wrapper.cu before %.cu
%_wrapper.cu:%.c
	cat $(P4A_ACCEL_DIR)/p4a_accel_wrapper-CUDA.h $< > $@

%_wrapper.cl:%.c
	cat $(P4A_ACCEL_DIR)/p4a_accel_wrapper-OpenCL.h $< > $@

%.cu:%.c
	ln -s $< $@ 

BASEFLAGS = -I.. -I.  -DP4A_PROFILING  -DUNIX 
#Flags for openMP mode
CFLAGS = $(BASEFLAGS) -DP4A_ACCEL_OPENMP -std=c99

#Flags for OpenCL mode 
CLFLAGS = $(BASEFLAGS) -DP4A_ACCEL_OPENCL -I$(CUDA_DIR)/include/CL -I$(CUDA_DIR)/include -std=c99 
LDFLAGS = -fPIC -L/usr/lib 
LDLIBS =  -lOpenCL 

#Flags for Cuda mode (nvcc ~ g++)
CPPFLAGS = $(BASEFLAGS) -DP4A_ACCEL_CUDA -I../../../../P4A_CUDA -I$(CUDA_DIR)/include 
CUFLAGS += --compiler-options -fno-strict-aliasing -O2 


# New default rule to compile CUDA source files:

$(EXECUTABLE): $(CFILES) $(WRAPFILES)
	$(CC) $(CFLAGS) -o $@ $(CFILES)  $(WRAPFILES)  

$(EXECUTABLE-OMP): $(CFILES) $(WRAPFILES)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(CFILES)  $(WRAPFILES) -fopenmp

# New default rule to compile CUDA source files:
%.cu.o: %.cu
	$(CUC) $(CPPFLAGS) $(CUFLAGS) -o $@ -c $< 

$(EXECUTABLE-CU): $(OBJCUFILES)
	$(CUC) -o $(EXECUTABLE-CU) $(OBJCUFILES) $(LDLIBS) 

# rule to compile OpenCL source files:
$(EXECUTABLE-CL): $(CFILES) $(CLFILES)
	$(CC) $(CLFLAGS) -o $@ $(CFILES) $(LDLIBS) $(LDFLAGS)

clean::
	rm -f $(EXECUTABLES) $(WRAPFILES) *.o $(CUFILES) $(CLFILES) *~ ./p4a_accel.c
