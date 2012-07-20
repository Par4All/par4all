TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# C99 mode ?
ifdef C99_MODE
P4A_FLAGS+= --c99
P4A_CUDA_SUFFIX=.c
else
P4A_CUDA_SUFFIX=.cu
endif 


# Common source file (for timing)
COMMON := $(TOP)/common/timing.c
ACCEL_FLAGS := -I$(P4A_ACCEL_DIR) -DP4A_ACCEL_CUDA 
OPENCL_ACCEL_FLAGS := -I$(P4A_ACCEL_DIR) -DP4A_ACCEL_OPENCL

ifndef CUDACC
ACCEL_FLAGS+=-arch=sm_20
else
P4A_FLAGS+=--cuda-cc=$(CUDACC)
endif

ACCEL_SRC := $(P4A_ACCEL_DIR)/p4a_accel.cu $(P4A_ACCEL_DIR)/p4a_communication_optimization_runtime.cpp
OPENCL_ACCEL_SRC := $(P4A_ACCEL_DIR)/p4a_accel.c
# Params for benchmarks
NRUNS := `seq 1 5`

# source file for the different version of the code
SOURCE := $(LOCAL_TARGET).c
OMP_SOURCE := $(LOCAL_TARGET).openmp.c
CUDA_SOURCE := $(LOCAL_TARGET).naive$(P4A_CUDA_SUFFIX)
CUDA_OPT_SOURCE := $(LOCAL_TARGET).opt$(P4A_CUDA_SUFFIX)
GENERATED_KERNELS = $(wildcard $(LOCAL_TARGET)_p4a.generated/*.cu)
OPENCL_SOURCE := $(LOCAL_TARGET).opencl.c
HMPP_SOURCE := hmpp/$(LOCAL_TARGET).hmpp.c
PPCG_PRE_SOURCE :=$(LOCAL_TARGET).ppcg.c
PPCG_GENERATED= *_host.cu *_kernel.cu


SEQ_TARGET := $(LOCAL_TARGET)_seq
OMP_TARGET := $(LOCAL_TARGET)_openmp
CUDA_TARGET := $(LOCAL_TARGET)_cuda
CUDA_OPT_TARGET := $(LOCAL_TARGET)_cuda_opt
OPENCL_TARGET := $(LOCAL_TARGET)_opencl
PGI_TARGET := $(LOCAL_TARGET)_cuda_pgi
HMPP_TARGET := $(LOCAL_TARGET)_cuda_hmpp
PPCG_TARGET := $(LOCAL_TARGET)_ppcg

# Compilation flags
CC := gcc
COMMON_FLAGS:= -I$(TOP)/common/ -DPOLYBENCH_TIME
CFLAGS+= -O3
GCCFLAGS:= -fno-strict-aliasing -fPIC -std=c99  
LDFLAGS:= -lm
P4A_FLAGS+= -p $(LOCAL_TARGET)_p4a -r
NVCC_FLAGS+= --compiler-options "-O3"

ifdef threads
NVCC_FLAGS+= -DP4A_CUDA_THREAD_MAX=$(threads)
endif

ifdef debug
GCCFLAGS+=-W -Wall -DP4A_DEBUG
PGI_FLAGS+=-Minfo
endif


# Default target : display usage
.PHONY: default clean dist-clean

default:
	more $(TOP)/README.txt

$(LOCAL_TARGET):
	more $(TOP)/README.txt


# build source for parallel versions
$(OMP_SOURCE): $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS) $(P4A_FLAGS) $(SIZE_PARAMS)
	mv $(<:%.c=%.p4a.c) $@
openmp_src:$(OMP_SOURCE)

$(CUDA_SOURCE): $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS) $(P4A_FLAGS) $(SIZE_PARAMS) --cuda 
	mv $(<:%.c=%.p4a$(P4A_CUDA_SUFFIX)) $@
cuda_src:$(CUDA_SOURCE)

$(CUDA_OPT_SOURCE): $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS) $(P4A_FLAGS) $(SIZE_PARAMS) --cuda --com-optimization
	mv $(<:%.c=%.p4a$(P4A_CUDA_SUFFIX)) $@
cuda_opt_src:$(CUDA_OPT_SOURCE)

$(OPENCL_SOURCE): $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS) $(P4A_FLAGS) $(SIZE_PARAMS) --opencl
	mv $(<:%.c=%.p4a.c) $@
opencl_src:$(OPENCL_SOURCE)


$(PPCG_PRE_SOURCE):  $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS) $(P4A_FLAGS) $(SIZE_PARAMS) -S -DPGCC
	mv $(SOURCE:%.c=%.p4a.c) $@


# build binary for the 3 different versions
$(SEQ_TARGET): $(SOURCE) $(COMMON)
	gcc $(COMMON_FLAGS) $(CFLAGS) $(GCCFLAGS) $(LDFLAGS) $(SIZE_PARAMS) $^ -o $@
seq: $(SEQ_TARGET)
$(LOCAL_TARGET)-seq:seq

$(OMP_TARGET): $(OMP_SOURCE) $(COMMON)
	gcc -fopenmp $(COMMON_FLAGS) $(CFLAGS) $(GCCFLAGS) $(LDFLAGS) $(SIZE_PARAMS) $^ -o $@
openmp: $(OMP_TARGET)
$(LOCAL_TARGET)-openmp:openmp

$(CUDA_TARGET): $(CUDA_SOURCE) $(COMMON)
	nvcc $^ -o $@ $(NVCC_FLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(ACCEL_FLAGS) $(ACCEL_SRC) $(GENERATED_KERNELS) $(SIZE_PARAMS)
cuda: $(CUDA_TARGET)
$(LOCAL_TARGET)-cuda:cuda

$(CUDA_OPT_TARGET): $(CUDA_OPT_SOURCE) $(COMMON)
	nvcc $^ -o $@ $(NVCC_FLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(ACCEL_FLAGS) $(ACCEL_SRC) $(GENERATED_KERNELS) $(SIZE_PARAMS)
cuda_opt: $(CUDA_OPT_TARGET)
$(LOCAL_TARGET)-cuda-opt:cuda_opt

$(OPENCL_TARGET): $(OPENCL_SOURCE) $(COMMON)
	@if [ -z "$(OPENCL_FLAGS)" ] ; then echo "It seems you didn't set $(OPENCL_FLAGS) variable" ; fi
	gcc -std=gnu99 $^ -o $@ $(COMMON_FLAGS) $(LDFLAGS) $(OPENCL_ACCEL_FLAGS) $(OPENCL_ACCEL_SRC) $(SIZE_PARAMS) $(OPENCL_FLAGS)
opencl: $(OPENCL_TARGET)
$(LOCAL_TARGET)-opencl:opencl

$(PGI_TARGET): $(SOURCE) $(COMMON)
	pgcc $^ -o $@ $(COMMON_FLAGS) $(LDFLAGS) $(CFLAGS) -DPGI_ACC -ta=nvidia,cc13 $(PGI_FLAGS) $(SIZE_PARAMS)
pgi: $(PGI_TARGET)
$(HMPP_TARGET): $(HMPP_SOURCE) $(COMMON)
	hmpp --codelet-required --nvcc-options -arch,sm_13 gcc $(COMMON_FLAGS) $(CFLAGS) $(GCCFLAGS) $(LDFLAGS) $^ -o $@ $(SIZE_PARAMS)
hmpp: $(HMPP_TARGET)
$(LOCAL_TARGET)-hmpp:hmpp

$(PPCG_TARGET): $(PPCG_PRE_SOURCE) $(COMMON)
	ppcg $(PPCG_PRE_SOURCE)
	nvcc -o $@ $(NVCC_FLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(ACCEL_FLAGS) $(ACCEL_SRC) $(PPCG_GENERATED) $(COMMON)
ppcg: $(PPCG_TARGET)
$(LOCAL_TARGET)-%:ppcg

# Run target
run_seq: $(SEQ_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done
run_openmp: $(OMP_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done
run_cuda: $(CUDA_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done
run_cuda_opt: $(CUDA_OPT_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done
run_opencl: $(OPENCL_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done
run_pgi: $(PGI_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done
run_hmpp: $(HMPP_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done
run_ppcg: $(PPCG_TARGET)
	for run in $(NRUNS); do \
		BENCH_SUITE=$(BENCH_SUITE) $(TOP)/scripts/record_measure.sh $(LOCAL_TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(LOCAL_TARGET)_$@.time`; \
	done


# Clean targets
clean: 
	rm -Rf $(OMP_SOURCE) $(CUDA_SOURCE) $(CUDA_OPT_SOURCE) *.database *.build .*.tmp p4a_new_files P4A  *.o mycodelet*.cu* $(PPCG_GENERATED) $(PPCG_PRE_SOURCE)
dist-clean: clean 
	rm -f $(LOCAL_TARGET) $(PGI_TARGET) $(HMPP_TARGET) $(CUDA_TARGET) $(CUDA_OPT_TARGET) $(OMP_TARGET) $(SEQ_TARGET) $(LOCAL_TARGET)_run_seq.time $(LOCAL_TARGET)_run_openmp.time $(LOCAL_TARGET)_run_cuda.time $(LOCAL_TARGET)_run_cuda_opt.time mycodelet*.so


