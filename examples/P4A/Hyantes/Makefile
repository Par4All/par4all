# A GNU makefile to run a demo of Par4All on Hyantes application 

LOCAL_TARGET= hyantes-static-99
LDLIBS+= -lm

SOURCES= $(LOCAL_TARGET:=.c)

RUN_ARG=Rhone-alpesXYLongLat_pop.txt 1.1 32 4 35 0.01 40 > output_$*

display% : run%
	# Display graphically the results:
	#gnuplot -persist -e "set pm3d map; set palette gray; set size ratio 1; splot 'output$*'"
	gnuplot -persist -e "set pm3d map; set size ratio 1; splot 'output$*'"

display_cuda-opt:
	@echo "Cuda optimisation is not yet supported"

include ../common.mk
