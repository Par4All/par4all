# A GNU makefile to run a demo of Par4All on "stars-pm" application
NVCC= nvcc
CC= gcc
TARGET= stars-pm

ifneq ($(do_not_cheat),1)
P4A_CHEAT= -DP4A_CUDA_CHEAT
endif

P4A_OPTIONS+=-DP4A
NVCCFLAGS=-lcufft

ifeq "$(USE_FLOAT)" "1"
# Use single precision for the OpenMP and GPU version
# Use also single precision for the sequential version:
CPPFLAGS+= -DUSE_FLOAT
endif


# can be overwrite on cmdline
SIZE=128
BIN_SUFFIX=_$(SIZE)

# You can set P4A_OPTIONS to pass options to p4a
CFLAGS+= -O3 -std=c99 -Wall
CPPFLAGS+= -D_GNU_SOURCE -I./include -I./  -DNP=$(SIZE)
LDLIBS+= -lm -lfftw3f
CULIBS+= -lm --fftw3
CPPFLAGS+=




ifeq ($(oldgpu),1)
CPPFLAGS+= -DCPUHISTO
else
NVCCFLAGS+=-arch sm_11 
endif



ifeq ($(debug),1)
CPPFLAGS+= -DP4A_DEBUG
BIN_SUFFIX:=$(BIN_SUFFIX)_dgb
endif

ifeq ($(timing),1)
GPU_TIMING=-DP4A_TIMING 
CPU_TIMING=-DCPU_TIMING
BIN_SUFFIX:=$(BIN_SUFFIX)_timing
endif

ifeq ($(bench),1)
CPPFLAGS+= -DP4A_BENCH
endif


# To play with OpenMP parameters:
#OMP_SCHEDULE="dynamic"
#OMP_NUM_THREADS=2
#OMP_DYNAMIC=false
#OMP_NESTED=false

#export OMP_SCHEDULE
#export OMP_NUM_THREADS
#export OMP_DYNAMIC
#export OMP_NESTED


COMMON_SOURCES = common/io.c

BASE_SOURCES= sequential/pm.c \
							sequential/1-discretization.c sequential/2-histogramme.c sequential/3-potential.c \
							sequential/4-updateforce.c sequential/4-updatevel.c sequential/6-updatepos.c

CUDA_SOURCES= cuda/pm.cu cuda/kernel_tools.cu \
							cuda/1-discretization.cu cuda/2-histogramme.cu cuda/3-potential.cu \
							cuda/4-updateforce.cu cuda/4-updatevel.cu cuda/6-updatepos.cu

CPROTO_GRAPHICS_SRC = common/graphics.c common/glgraphics.c

STUBS = stubs/pips_stubs.c


OBJS=$(SOURCES:.c=.o)

#.PHONY: clean demo $(addprefix demo_, $(TARGET)) $(addprefix run_, $(TARGET) $(TARGET:=_omp) $(TARGET:=_p4a_omp) $(TARGET:=_p4a_cuda))
#.DEFAULT:
#	echo Target $@ not implemented...

# Keep intermediate files for the demo for further inspection:
.PRECIOUS: $(TARGET:=.p4a.c) $(TARGET:=.p4a.cu) $(TARGET:=_seq) $(TARGET:=_openmp) $(TARGET:=_cuda) common/graphics.o


#.INTERMEDIATE: $(TARGET:=_seq)

ifeq ($(opengl),1)
GRAPHICS_CPPFLAGS+= -D_GLGRAPHICS_
CPPFLAGS+=$(GRAPHICS_CPPFLAGS)
GRAPHICS+= common/glgraphics.o
GRAPHICS_SRC+= common/glgraphics.c
LDLIBS+= -lGL -lGLU -lglut
CULIBS+= -lGL -lGLU -lglut
P4A_OPTIONS+= -D_GLGRAPHICS_ --extra-obj=common/glgraphics.o
BIN_SUFFIX:=$(BIN_SUFFIX)_opengl
endif
ifeq ($(gtk),1)
GRAPHICS_CPPFLAGS+= -D_GRAPHICS_ `pkg-config --cflags-only-I gtk+-2.0` 
CFLAGS+=`pkg-config --cflags gtk+-2.0`
CPPFLAGS+=-D_GRAPHICS_ 
GRAPHICS+= common/graphics.o
GRAPHICS_SRC+= common/graphics.c
LDLIBS+= `pkg-config --libs-only-l --libs-only-L gtk+-2.0`
CULIBS+= `pkg-config --libs-only-l --libs-only-L gtk+-2.0`
P4A_OPTIONS+= -D_GRAPHICS_ --extra-obj=common/graphics.o
BIN_SUFFIX:=$(BIN_SUFFIX)_gtk
endif

default:
	echo "This the content of the file README.txt:"
	# Use more and not less because when quitting, the displayed text
	# remains displayed...
	more README.txt

demo : display_seq display_openmp display_cuda ;

build-all: stars-pm_seq stars-pm_openmp stars-pm_autocuda \
						stars-pm_autocuda_comm_optimization stars-pm_cuda

clean :
	rm -rf $(TARGET:=_seq) $(TARGET:=_openmp) $(TARGET:=_cuda) \
				 $(TARGET:=.p4a.c) $(STUBS:.c=.p4a.c) \
				 $(TARGET:=.p4a.cu) $(STUBS:.c=.p4a.cu) \
				 $(COMMON_SOURCES:.c=.p4a.c)  $(BASE_SOURCES:.c=.p4a.c) $(GRAPHICS:.c=.p4a.c) \
				 $(COMMON_SOURCES:.c=.p4a.cu) $(BASE_SOURCES:.c=.p4a.cu) $(GRAPHICS:.c=.p4a.cu) \
				 $(CPROTO_GRAPHICS_SRC:.c=.o) \
				  *~ *.database *.build *.o include/stars-pm-generated*.h

run-%: $(TARGET)_%
	# Run a version and display timing information:
	time ./$< exp128.a.bin
	# It was total time (starting time, I/O and computations)


# To have shortcut as seq for of typing hyantes-static-99_seq
%:$(TARGET)_% ;

%_seq : include/stars-pm-generated_$(SIZE).h $(COMMON_SOURCES) $(BASE_SOURCES) $(GRAPHICS)
	# Compilation of the sequential program:
	$(CC) $(CPPFLAGS) $(CPU_TIMING) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@$(BIN_SUFFIX) $(COMMON_SOURCES) $(BASE_SOURCES) $(GRAPHICS)

%_openmp : include/stars-pm-generated_$(SIZE).h $(COMMON_SOURCES) $(BASE_SOURCES) $(GRAPHICS) $(STUBS)
	# Parallelize and build an OpenMP version:
	p4a -DFFTW3_THREADED -lfftw3f_threads $(P4A_OPTIONS) $(CPU_TIMING) $(CPPFLAGS)  -o $@$(BIN_SUFFIX) $(COMMON_SOURCES) $(BASE_SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.c) $(LDLIBS)
	# P4A openmp end !

%_autocuda : include/stars-pm-generated_$(SIZE).h $(COMMON_SOURCES) $(BASE_SOURCES) $(GRAPHICS) $(STUBS)
	# Parallelize and build a CUDA version:
	p4a $(P4A_OPTIONS) $(GPU_TIMING) $(P4A_CHEAT) $(CPPFLAGS)  --cuda -o $@$(BIN_SUFFIX) $(COMMON_SOURCES) $(BASE_SOURCES) $(STUBS)  --exclude-file=$(STUBS:.c=.p4a.cu) --exclude-file=$(STUBS:.c=.p4a.c) $(CULIBS)
	
%_autocuda_comm_optimization : include/stars-pm-generated_$(SIZE).h $(COMMON_SOURCES) $(BASE_SOURCES) $(GRAPHICS) $(STUBS)
	# Parallelize and build a CUDA version:
	p4a $(P4A_OPTIONS) $(GPU_TIMING) $(P4A_CHEAT) $(CPPFLAGS) --com-optimization --cuda -o $@$(BIN_SUFFIX) $(COMMON_SOURCES) $(BASE_SOURCES) $(STUBS)  --exclude-file=$(STUBS:.c=.p4a.cu) --exclude-file=$(STUBS:.c=.p4a.c) $(CULIBS)
	

%_cuda : include/stars-pm-generated_$(SIZE).h $(COMMON_SOURCES) $(CUDA_SOURCES) $(GRAPHICS)
	# Parallelize and build a CUDA version:
	$(NVCC) $(CPPFLAGS) $(GPU_TIMING) $(LDFLAGS) $(NVCCFLAGS) $(LDLIBS) -o $@$(BIN_SUFFIX) $(COMMON_SOURCES) $(CUDA_SOURCES) $(GRAPHICS)

	
include/stars-pm-generated_$(SIZE).h :  $(COMMON_SOURCES) $(BASE_SOURCES) $(CPROTO_GRAPHICS_SRC)
	@echo "Generating headers $@" 
	echo >$@
	rm -f include/stars-pm-generated_* common/*.o *.o
	cproto  `pkg-config --cflags-only-I gtk+-2.0` $(COMMON_SOURCES) $(BASE_SOURCES) $(CPROTO_GRAPHICS_SRC) $(CPPFLAGS) $(GRAPHICS_CPPFLAGS) -I./include/ > $@
	
	  