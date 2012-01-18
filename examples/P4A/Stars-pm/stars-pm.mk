# Makefile for Stars-pm
LOCAL_TARGET= stars-pm

# can be overwrite on cmdline
SIZE?=128

# In the following, the _GRAPHICS_ and CPP flags allow GTK or OpenGL
# graphics output. Note you can have both at the same time,
# interessingly. :-)

#colors
NO_COLOR=

RUN_ARG=./data/exp$(SIZE).a.bin

COMMON_SOURCES = common/io.c
COMMON_INCLUDES = include/stars-pm-generated_$(SIZE).h
COMMON_GRAPHICS_SRC = common/glgraphics.c common/graphics.c
CLEAN_OTHERS = $(COMMON_INCLUDES) $(COMMON_GRAPHICS_SRC:.c=.o)


BASE_SOURCES= pm.c 1-discretization.c 2-histogramme.c 3-potential.c \
	4-updateforce.c 4-updatevel.c 6-updatepos.c
OBJS=$(BASE_SOURCES:.c=.o)

SOURCES= $(BASE_SOURCES:%=sequential/%)
ifndef P4A_GENERATED
  P4A_GENERATED=$(SOURCES:%.c=%.p4a.c) $(SOURCES:%.c=%.p4a.cu)
endif
export P4A_GENERATED
PGI_SOURCES=  $(BASE_SOURCES:%=pgi/%)
PGI_OBJS=  $(BASE_SOURCES:%.c=%.o)
MANUAL_CUDA_SOURCES= cuda/pm.cu cuda/kernel_tools.cu \
	cuda/1-discretization.cu cuda/2-histogramme.cu cuda/3-potential.cu \
	cuda/4-updateforce.cu cuda/4-updatevel.cu cuda/6-updatepos.cu

STUBS = stubs/pips_stubs.c

#Graphics
GRAPHICS_OBJ = $(GRAPHICS_SRC:.c=.o)
CPROTO_GRAPHICS_SRC = $(GRAPHICS_SRC)

# Compilation flags
P4A_OMP_FLAGS=-DFFTW3_THREADED -lfftw3f_threads
P4A_CUDA_FLAGS=--atomic
P4A_ACCEL_OPENMP_FLAGS=$(P4A_OMP_FLAGS)
NVCCFLAGS+=-lcufft
LDLIBS+= -lm -lfftw3f
CULIBS+= -lm --fftw3
OPENCLLIBS+= -lm -lfftw3f
CPPFLAGS+= -DNP=$(SIZE) -I./include -I./ -D_GNU_SOURCE
CFLAGS+= --std=gnu99

ifeq ($(opengl),1)
  GRAPHICS_CPPFLAGS+= -D_GLGRAPHICS_
  CPPFLAGS+=$(GRAPHICS_CPPFLAGS)
  GRAPHICS_SRC+= common/glgraphics.c
  LDLIBS+= -lGL -lGLU -lglut
  CULIBS+= -lGL -lGLU -lglut
  OPENCLLIBS+= -lGL -lGLU -lglut
  # Since the user may have defined its own P4A_OPTIONS such as
  # P4A_OPTIONS='--cuda-cc=1.3', use it and go on with it with 'overide':
  override P4A_OPTIONS+= -D_GLGRAPHICS_ --extra-obj=common/glgraphics.o
  BIN_SUFFIX:=$(BIN_SUFFIX)_opengl
endif

ifeq ($(gtk),1)
  GRAPHICS_CPPFLAGS+= -D_GRAPHICS_ `pkg-config --cflags-only-I gtk+-2.0`
  CFLAGS+=`pkg-config --cflags gtk+-2.0`
  CPPFLAGS+= -D_GRAPHICS_
  GRAPHICS_SRC+= common/graphics.c
  LDLIBS+= `pkg-config --libs-only-l --libs-only-L gtk+-2.0`
  CULIBS+= `pkg-config --libs-only-l --libs-only-L gtk+-2.0`
  OPENCLLIBS+=`pkg-config --libs-only-l --libs-only-L gtk+-2.0`
  override P4A_OPTIONS+= -D_GRAPHICS_ --extra-obj=common/graphics.o
  BIN_SUFFIX:=$(BIN_SUFFIX)_gtk
endif

include/stars-pm-generated_$(SIZE).h :  $(COMMON_SOURCES) $(SOURCES) $(CPROTO_GRAPHICS_SRC)
	@echo "Generating headers $@"
	echo >$@
	rm -f include/stars-pm-generated_* *.o
	cproto  `pkg-config --cflags-only-I gtk+-2.0` $(COMMON_SOURCES) $(SOURCES) $(CPROTO_GRAPHICS_SRC) $(CPPFLAGS) $(GRAPHICS_CPPFLAGS) -I./include/ > $@

opengl_display_%: demo_opengl_message run_%_opengl ;

demo_opengl_message:
	@echo "\n Please close the window at the end of the execution. In case of the chain of demos \
	(command "make demo_opengl"), this allows the execution of the next demo\n"

demo_opengl : opengl_display_seq opengl_display_openmp opengl_display_accel-openmp opengl_display_cuda opengl_display_cuda-opt opengl_display_opencl;


# Changing between OpenGL and GTK still needs a make clean in between...

# If we ask for GTK version, compile as is:
%_gtk:
	$(MAKE) gtk=1 $*


# If we ask for OpenGL version, compile as is:
%_opengl: demo_opengl_message
	$(MAKE) opengl=1 $*

default_gtk=
ifneq ($(gtk),1)
  ifneq ($(opengl),1)
	# Default display use GTK version if not asked already for OpenGL
	#nor GTK:
	default_gtk=_gtk
  endif
endif

display_%: run_%$(default_gtk) ;
