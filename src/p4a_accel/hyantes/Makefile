# Add source files here
TARGET = hyantes
CFILES = hyantes-p4a.c
KERNELFILES = kernel_wrapper.c
LDLIBS = -lm
BASEFLAGS = -DUSE_FLOAT

P4A_ACCEL_DIR = ..
include $(P4A_ACCEL_DIR)/p4a_accel.mk

run-seq: 
	time hyantes-seq Rhone-alpesXYLongLat_pop.txt 1.1 32 0.01 40 > h.out 

run-omp: 
	time hyantes-omp Rhone-alpesXYLongLat_pop.txt 1.1 32 0.01 40 > h.out

run-cuda: 
	time hyantes-cuda Rhone-alpesXYLongLat_pop.txt 1.1 32 0.01 40 > h.out

run-cl: 
	time hyantes-cl Rhone-alpesXYLongLat_pop.txt 1.1 32 0.01 40 > h.out

display:
	gnuplot -persist -e "set pm3d map; set size ratio 1; splot 'h.out'"

