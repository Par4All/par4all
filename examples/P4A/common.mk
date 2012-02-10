# A common GNU makefile to run a demo of Par4All on example application
# LOCAL_TARGET, SOURCES, OBJS, and many others should be defined

NVCC= nvcc
CC= gcc
PGCC= pgcc
P4A_OPTIONS+=-DP4A

ifeq "$(USE_FLOAT)" "1"
# Use single precision for the OpenMP and GPU version
# Use also single precision for the sequential version:
CPPFLAGS+= -DUSE_FLOAT
endif


# You can set P4A_OPTIONS to pass options to p4a
CFLAGS+= -O3 -std=c99 -Wall
CPPFLAGS+= -D_GNU_SOURCE
CPPFLAGS+=


ifeq ($(debug),1)
CPPFLAGS+= -DP4A_DEBUG
endif

ifeq ($(bench),1)
CPPFLAGS+= -DP4A_BENCH
endif




# Keep intermediate files for the demo for further inspection:
.PRECIOUS: $(LOCAL_TARGET:=.p4a.c) $(LOCAL_TARGET:=.p4a.cu) $(LOCAL_TARGET:=.p4a-accel.cu) $(LOCAL_TARGET:=-seq) $(LOCAL_TARGET:=-openmp) $(LOCAL_TARGET:=-cuda) $(LOCAL_TARGET:=-accel-openmp) $(LOCAL_TARGET:=-opencl)

.DEFAULT_GOAL := default

default:
	echo "This the content of the file README.txt:"
	# Use more and not less because when quitting, the displayed text
	# remains displayed...
	more ../README.txt README.txt

demo : display_seq display_openmp display_cuda display_cuda-opt display_accel-openmp display_opencl;

build-all: $(LOCAL_TARGET)-seq $(LOCAL_TARGET)-openmp $(LOCAL_TARGET)-pgi $(LOCAL_TARGET)-accel-openmp \
						$(LOCAL_TARGET)-cuda $(LOCAL_TARGET)-cuda-opt $(LOCAL_TARGET)-cuda-manual $(LOCAL_TARGET)-opencl

clean :
	rm -rf $(LOCAL_TARGET)-seq $(LOCAL_TARGET)-openmp $(LOCAL_TARGET)-pgi $(LOCAL_TARGET)-accel-openmp \
				 $(LOCAL_TARGET)-cuda $(LOCAL_TARGET)-cuda-opt $(LOCAL_TARGET)-cuda-manual $(LOCAL_TARGET)-opencl \
				 $(LOCAL_TARGET:=.p4a.c) $(STUBS:.c=.p4a.c) \
				 $(LOCAL_TARGET:=.p4a.cu) $(STUBS:.c=.p4a.cu) \
				 $(COMMON_SOURCES:.c=.p4a.c) $(SOURCES:.c=.p4a.c) \
				 $(COMMON_SOURCES:.c=.p4a.cu) $(SOURCES:.c=.p4a.cu) \
				 $(CLEAN_OTHERS) \
				 *~ *.database *.build *.o p4a_new_files *.generated *.cl \
	output_accel-openmp output_cuda output_cuda-opt \
	output_opencl output_openmp output_seq


run_%: $(LOCAL_TARGET)-%
	# Run a version and display timing information:
	time ./$< $(RUN_ARG)
	# It was total time (starting time, I/O and computations)


# To have shortcut as seq for of typing hyantes-static-99_seq
%:$(LOCAL_TARGET)-% ;

$(LOCAL_TARGET)-seq : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(SOURCES)
	# Compilation of the sequential program:
	$(CC) $(CPPFLAGS) $(CPU_TIMING) $(CFLAGS) $(LDFLAGS) -o $@ $(COMMON_SOURCES) $(SOURCES) $(GRAPHICS_SRC) $(LDLIBS)

$(LOCAL_TARGET)-pgi : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(PGI_SOURCES)
	# Parallelize and build a CUDA version using PGI accelerator
	$(PGCC) -ta=nvidia,time $(CPPFLAGS) $(CPU_TIMING) $(LDFLAGS) -o $@ $(COMMON_SOURCES) $(PGI_SOURCES) $(LDLIBS)

$(LOCAL_TARGET)-openmp : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(SOURCES) $(STUBS) $(GRAPHICS_OBJ)
	# Parallelize and build an OpenMP version:
	p4a $(P4A_OMP_FLAGS) $(P4A_OPTIONS) $(CPU_TIMING) $(CPPFLAGS) -o $@ $(COMMON_SOURCES) $(SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.c) $(LDLIBS)
	# P4A openmp end !

$(LOCAL_TARGET)-accel-openmp : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(SOURCES) $(STUBS) $(GRAPHICS_OBJ)
	# Parallelize and build an OpenMP version:
	p4a -A --openmp $(P4A_ACCEL_OPENMP_FLAGS) $(P4A_OPTIONS) $(CPU_TIMING) $(CPPFLAGS) -o $@ $(COMMON_SOURCES) $(SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.c) $(LDLIBS)
	# P4A openmp end !

$(LOCAL_TARGET)-cuda : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(SOURCES) $(STUBS) $(GRAPHICS_OBJ)
	# Parallelize and build a CUDA version:
	p4a $(P4A_CUDA_FLAGS) $(P4A_OPTIONS) $(GPU_TIMING) $(CPPFLAGS) --cuda -o $@ $(COMMON_SOURCES) $(SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.cu) --exclude-file=$(STUBS:.c=.p4a.c) $(CULIBS) --nvcc-flags="$(NVCCFLAGS)"

$(LOCAL_TARGET)-opencl : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(SOURCES) $(STUBS) $(GRAPHICS_OBJ)
	# Parallelize and build an OpenCL version:
	p4a $(P4A_OPENCL_FLAGS) $(P4A_OPTIONS) $(GPU_TIMING) $(CPPFLAGS) --opencl -o $@ $(COMMON_SOURCES) $(SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.cu) --exclude-file=$(STUBS:.c=.p4a.c) $(OPENCLLIBS)

$(LOCAL_TARGET)-cuda-opt : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(SOURCES) $(STUBS) $(GRAPHICS_OBJ)
	# Parallelize and build a CUDA version:
	p4a $(P4A_CUDA_FLAGS) $(P4A_OPTIONS) $(GPU_TIMING) $(CPPFLAGS) --com-optimization --cuda -o $@ $(COMMON_SOURCES) $(SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.cu) --exclude-file=$(STUBS:.c=.p4a.c) $(CULIBS) --nvcc-flags="$(NVCCFLAGS)"

$(LOCAL_TARGET)-cuda-manual : $(COMMON_INCLUDES) $(COMMON_SOURCES) $(MANUAL_CUDA_SOURCES) $(GRAPHICS_OBJ)
	# Parallelize and build a CUDA version:
	$(NVCC) $(CPPFLAGS) $(GPU_TIMING) $(LDFLAGS) $(NVCCFLAGS) -o $@ $(COMMON_SOURCES) $(MANUAL_CUDA_SOURCES) $(LDLIBS)
