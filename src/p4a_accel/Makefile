# Use the PIPS infrastructure to generate a Doxygen documentation

# make clean
# make doxygen
# make doxygen-publish

# Some own Par4All stuff:
include $(P4A_ROOT)/doc/libs/par4all.mk

# common stuff, recycled directly from PIPS:
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= pips

# Some pure local Doxygen parameters:
define DOXYGEN_PARAMETERS
INPUT_ENCODING         = UTF-8
INPUT                  = p4a_accel-CUDA.c p4a_accel-CUDA.h \
	p4a_accel-OpenCL.c p4a_accel-OpenCL.h \
	p4a_accel-OpenMP.c p4a_accel-OpenMP.h \
	p4a_accel-shared.c p4a_accel.c p4a_accel.h \
	p4a_accel_wrapper-OpenCL.h \
	p4a_communication_optimization_runtime.cpp \
	p4a_fftw3_runtime.cpp \
	p4a_include-OpenCL.h \
	p4a_runtime_interface.f95 \
	p4a_stacksize_test.h \
	p4a_stubs.c p4a_stubs.f
endef

PROJECT_NAME           = Par4All_Accel_runtime


## Where we want the documentation to be published:
#PUBLISH_LOCATION := /tmp/$(PROJECT_NAME)

include $(ROOT)/makes/doxygen.mk
