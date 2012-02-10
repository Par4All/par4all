# A GNU makefile to run a demo of Par4All on Hyantes application

LOCAL_TARGET= hyantes-static-99
LDLIBS+= -lm

SOURCES= $(LOCAL_TARGET:=.c)

RUN_ARG=Rhone-alpesXYLongLat_pop.txt 1.1 32 4 35 0.01 40 > output_$*

include ../common.mk

display% : run%
	# Display graphically the results:
	#gnuplot -persist -e "set pm3d map; set palette gray; set size ratio 1; splot 'output$*'"
	gnuplot -persist -e "set pm3d map; set size ratio 1; splot 'output$*'"

hyantes-static-99-cuda-opt:
	@echo "CUDA communication optimisation is not yet supported here but it would not be interesting here anyway..."

run_cuda-opt: hyantes-static-99-cuda-opt
	# Do nothing

display_cuda-opt: hyantes-static-99-cuda-opt
	# Do nothing
