# Add source files here
TARGET = hyantes
CFILES = hyantes-p4a.c
KERNELFILES = kernel_wrapper.c
LDLIBS = -lm
BASEFLAGS = -DUSE_FLOAT -DP4A_PROFILING
#BASEFLAGS = -DUSE_FLOAT -DP4A_TIMING
DUMMYFILE = h.out

P4A_ACCEL_DIR = ../../..
include $(P4A_ACCEL_DIR)/p4a_accel.mk

run%: $(TARGET)%
	time ./$< ../Rhone-alpesXYLongLat_pop.txt 1.1 32 0.01 40 > $(DUMMYFILE)

display:
	gnuplot -persist -e "set pm3d map; set size ratio 1; splot 'h.out'"

