# Makefile for Stars-pm
TARGET= stars-pm

# can be overwrite on cmdline
SIZE?=128

RUN_ARG=exp$(SIZE).a.bin

COMMON_SOURCES = common/io.c
COMMON_INCLUDES = include/stars-pm-generated_$(SIZE).h
CPROTO_GRAPHICS_SRC = common/graphics.c common/glgraphics.c
CLEAN_OTHERS = $(COMMON_INCLUDES) $(CPROTO_GRAPHICS_SRC:.c=.o) \


BASE_SOURCES= pm.c 1-discretization.c 2-histogramme.c 3-potential.c \
	4-updateforce.c 4-updatevel.c 6-updatepos.c
OBJS=$(BASE_SOURCES:.c=.o)

SOURCES= $(BASE_SOURCES:%=sequential/%)
export P4A_GENERATED=$(SOURCES:%.c=%.p4a.c) $(SOURCES:%.c=%.p4a.cu)
PGI_SOURCES=  $(BASE_SOURCES:%=pgi/%)
PGI_OBJS=  $(BASE_SOURCES:%.c=%.o)
MANUAL_CUDA_SOURCES= cuda/pm.cu cuda/kernel_tools.cu \
							cuda/1-discretization.cu cuda/2-histogramme.cu cuda/3-potential.cu \
							cuda/4-updateforce.cu cuda/4-updatevel.cu cuda/6-updatepos.cu


STUBS = stubs/pips_stubs.c



# Compilation flags
P4A_OMP_FLAGS=-DFFTW3_THREADED -lfftw3f_threads 
P4A_CUDA_FLAGS=--atomic
P4A_ACCEL_OPENMP_FLAGS=$(P4A_CUDA_FLAGS)
NVCCFLAGS=-lcufft
LDLIBS+= -lm -lfftw3f
CULIBS+= -lm --fftw3
CPPFLAGS+= -DNP=$(SIZE) -I./include -I./ -D_GNU_SOURCE
CFLAGS+= --std=gnu99


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


include/stars-pm-generated_$(SIZE).h :  $(COMMON_SOURCES) $(SOURCES) $(CPROTO_GRAPHICS_SRC)
	@echo "Generating headers $@" 
	echo >$@
	rm -f include/stars-pm-generated_* common/*.o *.o
	cproto  `pkg-config --cflags-only-I gtk+-2.0` $(COMMON_SOURCES) $(SOURCES) $(CPROTO_GRAPHICS_SRC) $(CPPFLAGS) $(GRAPHICS_CPPFLAGS) -I./include/ > $@
	
