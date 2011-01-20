# Add source files here
TARGET = jacobi 
#EXECUTABLE = jacobi 
CFILES = jacobi-p4a.c
KERNELFILES = kernel1_wrapper.c kernel2_wrapper.c

P4A_ACCEL_DIR = ../../..
include $(P4A_ACCEL_DIR)/p4a_accel.mk

run%: $(TARGET)%
	time ./$< ../Logo_HPC-Project-512x247-crop.pgm 

display:
	gimp output.pgm

