#
# Source, header and object files used to build the library.
LIB_CFILES= 	hyperplane.c  \
	 	hyperplane_direction.c scanning_base.c \
	 	global_parallelization.c  code_generation.c \
		tiling.c

LIB_HEADERS=	hyperplane-local.h

LIB_OBJECTS=  	hyperplane.o \
	 	hyperplane_direction.o scanning_base.o \
	 	global_parallelization.o code_generation.o \
		tiling.o




