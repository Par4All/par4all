#
# Source, header and object files used to build the library.
#

LIB_CFILES = 	hyperplane.c  \
	 	hyperplane_direction.c \
		scanning_base.c \
	 	global_parallelization.c  \
		code_generation.c \
		tiling.c \
		unimodular.c

LIB_HEADERS =	hyperplane-local.h

LIB_OBJECTS =  	$(LIB_CFILES:.c=.o)




