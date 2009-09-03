# Add source files here
EXECUTABLE	:= init
# Cuda source files (compiled with cudacc)
CUFILES		:= init.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:=

INCLUDES += -I..

COMMONFLAGS += --ptxas-options=-v  -arch=sm_13 --keep
# --clean

################################################################################
# Rules and targets

ROOTDIR=$(HOME)/NVIDIA_GPU_Computing_SDK/C/src

include $(ROOTDIR)/../common/common.mk
