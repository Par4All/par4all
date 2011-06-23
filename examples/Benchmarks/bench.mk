TOP := $(dir $(lastword $(MAKEFILE_LIST)))

# Common source file (for timing...)
COMMON := ../../common/timing.c
ACCEL_FLAGS := -I$(P4A_ACCEL_DIR) -DP4A_ACCEL_CUDA -arch=sm_13
ACCEL_SRC := $(P4A_ACCEL_DIR)/p4a_accel.cu $(P4A_ACCEL_DIR)/p4a_communication_optimization_runtime.cpp
# Params for benchmarks
NRUNS := `seq 1 3`

# source file for the different version of the code
SOURCE := $(TARGET).c
OMP_SOURCE := $(TARGET).openmp.c
CUDA_SOURCE := $(TARGET).naive.c
CUDA_OPT_SOURCE := $(TARGET).opt.c
GENERATED_KERNELS = $(wildcard p4a_new_files/*.cu)

SEQ_TARGET := $(TARGET)_seq
OMP_TARGET := $(TARGET)_openmp
CUDA_TARGET := $(TARGET)_cuda
CUDA_OPT_TARGET := $(TARGET)_cuda_opt


# Compilation flags
CC := gcc
COMMON_FLAGS:= -I../../common/ -DPOLYBENCH_TIME
CFLAGS:= -O3 -fno-strict-aliasing -fPIC -std=c99  
LDFLAGS:= -lm
P4A_FLAGS+= --c99 -p $(TARGET)_p4a -r

ifdef debug
CFLAGS+=-W -Wall -DP4A_DEBUG
endif

# Default target : display usage
.PHONY: default clean dist-clean

default:
	more $(TOP)/USAGE

$(TARGET):
	more $(TOP)/USAGE


# build source for parallel versions
$(OMP_SOURCE): $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS) $(P4A_FLAGS)
	mv $(<:%.c=%.p4a.c) $@

$(CUDA_SOURCE): $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS)  $(P4A_FLAGS) --cuda
	mv $(<:%.c=%.p4a.c) $@

$(CUDA_OPT_SOURCE): $(SOURCE) $(COMMON)
	p4a $^ $(COMMON_FLAGS) $(P4A_FLAGS) --cuda --com-optimization
	mv $(<:%.c=%.p4a.c) $@


# build binary for the 3 different versions
$(SEQ_TARGET): $(SOURCE) $(COMMON)
	gcc $(COMMON_FLAGS) $(CFLAGS) $(LDFLAGS) $^ -o $@
seq: $(SEQ_TARGET)

$(OMP_TARGET): $(OMP_SOURCE) $(COMMON)
	gcc -fopenmp $(COMMON_FLAGS) $(CFLAGS) $(LDFLAGS) $^ -o $@
openmp: $(OMP_TARGET)

$(CUDA_TARGET): $(CUDA_SOURCE) $(COMMON)
	nvcc $^ -o $@ $(COMMON_FLAGS) $(LDFLAGS) $(ACCEL_FLAGS) $(ACCEL_SRC) $(GENERATED_KERNELS)
cuda: $(CUDA_TARGET)
$(CUDA_OPT_TARGET): $(CUDA_OPT_SOURCE) $(COMMON)
	nvcc $^ -o $@ $(COMMON_FLAGS) $(LDFLAGS) $(ACCEL_FLAGS) $(ACCEL_SRC) $(GENERATED_KERNELS)
cuda_opt: $(CUDA_OPT_TARGET)

# Run target
run_seq: $(SEQ_TARGET)
	for run in $(NRUNS); do \
		$(TOP)/scripts/record_measure.sh $(TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(TARGET)_$@.time`; \
	done
run_openmp: $(OMP_TARGET)
	for run in $(NRUNS); do \
		$(TOP)/scripts/record_measure.sh $(TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(TARGET)_$@.time`; \
	done
run_cuda: $(CUDA_TARGET)
	for run in $(NRUNS); do \
		$(TOP)/scripts/record_measure.sh $(TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(TARGET)_$@.time`; \
	done
run_cuda_opt: $(CUDA_OPT_TARGET)
	for run in $(NRUNS); do \
		$(TOP)/scripts/record_measure.sh $(TARGET) $@ `./$< $(RUN_ARGS) | tee -a $(TARGET)_$@.time`; \
	done

# Clean targets
clean: 
	rm -Rf $(OMP_SOURCE) $(CUDA_SOURCE) $(CUDA_OPT_SOURCE) *.database *.build .*.tmp p4a_new_files
dist-clean: clean 
	rm -f $(TARGET) $(CUDA_TARGET) $(CUDA_OPT_TARGET) $(OMP_TARGET) $(SEQ_TARGET) $(TARGET)_run_seq.time $(TARGET)_run_openmp.time $(TARGET)_run_cuda.time $(TARGET)_run_cuda_opt.time


