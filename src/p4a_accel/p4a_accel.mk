CC         = gcc
CUC        = nvcc
LINK       = $(CC) -fPIC
CUDA_DIR   =/usr/local/cuda
STDDEFDIR  =/usr/local/par4all/packages/pips-gfc/gcc/ginclude

BASEFLAGS += -I$(P4A_ACCEL_DIR) -I.. -I. -DUNIX

ifdef debug
BASEFLAGS += -DP4A_DEBUG
endif

#Flags for openMP mode
CFLAGS = $(BASEFLAGS) -DP4A_ACCEL_OPENMP -std=c99 

#Flags for OpenCL mode 
ifdef P4A_OPENCL_INCLUDE_FLAGS
OPENCL_INC_FLAGS = $(P4A_OPENCL_INCLUDE_FLAGS)
else
OPENCL_INC_FLAGS = -I$(CUDA_DIR)/include/CL2 -I$(CUDA_DIR)/include/CL -I$(CUDA_DIR)/include
endif

ifdef P4A_OPENCL_LIBDIR_FLAGS
OPENCL_LIBDIR_FLAGS = $(P4A_OPENCL_LIBDIR_FLAGS)
else
OPENCL_LIBDIR_FLAGS = -L$(CUDA_DIR)/lib/lib  -L$(CUDA_DIR)/lib/lib64
endif

CLFLAGS = $(BASEFLAGS) -DP4A_ACCEL_OPENCL -std=c99 $(OPENCL_INC_FLAGS)
LDFLAGS = -fPIC -L/usr/lib  $(OPENCL_LIBDIR_FLAGS)
CLLIBS =  -lOpenCL 

#Flags for Cuda mode (nvcc ~ g++)
CPPFLAGS = $(BASEFLAGS) -DP4A_ACCEL_CUDA -I../../../../P4A_CUDA -I$(CUDA_DIR)/include 
CUFLAGS += --compiler-options -fno-strict-aliasing -O2 


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
	cpp $(CLFLAGS) -P $< -o $@

%.cu:%.c
	ln -s $< $@ 


# New default rule to compile CUDA source files:

$(EXECUTABLE): $(CFILES) $(KERNELFILES)
	$(CC) $(CFLAGS) -o $@ $(CFILES)  $(KERNELFILES) $(LDLIBS)  

$(EXECUTABLE-OMP): $(CFILES) $(KERNELFILES)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(CFILES)  $(KERNELFILES) $(LDLIBS)  -fopenmp

# New default rule to compile CUDA source files:
%.cu.o: %.cu
	$(CUC) $(CPPFLAGS) $(CUFLAGS) -o $@ -c $< 

$(EXECUTABLE-CU): $(OBJCUFILES)
	$(CUC) -o $(EXECUTABLE-CU) $(OBJCUFILES) $(LDLIBS) 

# rule to compile OpenCL source files:
$(EXECUTABLE-CL): $(CFILES) $(CLFILES)
	$(CC) $(CLFLAGS) -o $@ $(CFILES) $(CLLIBS) $(LDFLAGS) $(LDLIBS) 

clean::
	rm -f $(EXECUTABLES) *.o $(CUFILES) $(CLFILES) *~ ./p4a_accel.c $(DUMMYFILE)
