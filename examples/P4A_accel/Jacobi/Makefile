# A GNU makefile to run a demo of Par4All

TARGETS= jacobi

OPENMP= -fopenmp
CFLAGS= -O3 -std=c99
# -Wall
LDLIBS= -lm
DEMO_EDITOR=gedit

# This can be overriden by CUDA_DIR and CUDA_SDK environment variables:
CUDA_DIR=/usr/local/cuda
CUDA_SDK_DIR=$(HOME)/NVIDIA_GPU_Computing_SDK

# For debugging, add -g -DP4A_DEBUG
P4A_CFLAGS= -std=c99 -fopenmp -O3 -I$(P4A_ACCEL_DIR) -DP4A_ACCEL_OPENMP -I$(CUDA_DIR)/include -I$(CUDA_SDK_DIR)/C/common/inc

NVCC=$(CUDA_DIR)/bin/nvcc
# GT200
#CUDA_ARCHITECTURE= -arch=sm_13
# G92
CUDA_ARCHITECTURE= -arch=sm_11
CUDA_CFLAGS= --compiler-options -fno-strict-aliasing --ptxas-options=-v $(CUDA_ARCHITECTURE) -I$(P4A_ACCEL_DIR) -DP4A_ACCEL_CUDA -I$(CUDA_DIR)/include -I$(CUDA_SDK_DIR)/C/common/inc -DUNIX -O2
CUDA_LDFLAGS= -L$(CUDA_DIR)/lib64 -L$(CUDA_SDK_DIR)/C/lib -L$(CUDA_SDK_DIR)/C/common/lib/linux
CUDA_LIBS= -lcudart -lcutil_x86_64
LD_LIBRARY_PATH=$(CUDA_DIR)/lib64
export LD_LIBRARY_PATH

# To play with OpenMP parameters:
#OMP_SCHEDULE="dynamic"
#OMP_NUM_THREADS=2
OMP_DYNAMIC=false
OMP_NESTED=false

#export OMP_SCHEDULE
#export OMP_NUM_THREADS
export OMP_DYNAMIC
#export OMP_NESTED


SOURCES= $(TARGETS:=.c)

# Where to find files:
vpath %.h $(P4A_ACCEL_DIR)
vpath %.c $(P4A_ACCEL_DIR)

#.PHONY: clean demo $(addprefix demo_, $(TARGETS)) $(addprefix run_, $(TARGETS) $(TARGETS:=_omp) $(TARGETS:=_p4a_omp) $(TARGETS:=_p4a_cuda))

#.DEFAULT:
#	echo Target $@ not implemented...

# Keep intermediate files for the demo:
.PRECIOUS: $(TARGETS:=_omp.c) $(TARGETS:=.database)


.INTERMEDIATE: $(TARGETS)


default:
	echo "This the content of the file README.txt:"
	# Use more and not less because when quitting, the displayed text
	# remains displayed...
	more README.txt

demo_% : run_% run_%_omp run_%_p4a_omp run_%_p4a_cuda ;

# The full demo:
demo : $(addprefix demo_, $(TARGETS))


clean :
	rm -rf $(TARGETS) $(TARGETS:=_omp) $(TARGETS:=_omp.c) $(TARGETS:=_p4a_omp) $(TARGETS:=_p4a_cuda) *~ *.database output.pgm amplitude.pgm phase.pgm

all : $(TARGET) $(TARGET)_pips_omp

% : %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

# Use complex more solution to avoid interaction with .c rules...
run_fresne%:
	$(MAKE) fresne$*
	time ./fresne$* Logo_HPC-Project-GTC.pgm 100 amplitude.pgm phase.pgm
	# It was total time (starting time, I/O and computaions)
	# Compare the result with the reference:
	-diff -q amplitude.pgm amplitude-ref.pgm; diff -q phase.pgm phase-ref.pgm

run_jacob%:
	$(MAKE) jacob$*
	time ./jacob$* Logo_HPC-Project-GTC.pgm
	# It was total time (starting time, I/O and computations)
	# Compare the result with the reference:
	-diff -q output.pgm output-ref.pgm


# This must be after the previous rules to avoid run_jacobi_omp to be
# interpreted as a sequel of run_jacobi_omp.c... :-)

%_omp.c : %.c Makefile
	# Parallelize the code:
	pips_c2openmp $<
	# The result should be in the more recent PIPS database.
	cat `ls -td *.database | head -1`/Src/*.c > $@

%_omp : %_omp.c Makefile
	$(CC) $(CFLAGS) $(LDFLAGS) $(OPENMP) $(LDLIBS) -o $@ $<

# Build the binary from the output of PIPS:
%_p4a_omp : %.database Makefile
	@echo Compiling Par4All OpenMP version
	gcc $(P4A_CFLAGS) $*.database/P4A/*.c $(P4A_ACCEL_DIR)/p4a_accel.c -o $@ -lm

# Build the binary from the output of PIPS:
%_p4a_cuda : %.database Makefile
	@echo Compiling Par4All CUDA version
	cd $*.database/P4A; $(NVCC) -c $(CUDA_CFLAGS) *.cu $(P4A_ACCEL_DIR)/p4a_accel.cu
	cd $*.database/P4A; g++ -fPIC $(CUDA_LDFLAGS) *.o -o $@ $(CUDA_LIBS) -lm
	mv $*.database/P4A/$@ .

# Run PIPS
%.database : %.c
	#tpips $*.tpips
	python $*.py
