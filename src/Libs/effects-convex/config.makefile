#
# CFLAGS=		-g -ansi -Wall -mv8 -pipe
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=  interface.c old_utils.c old_prettyprint.c old_combine.c \
	old_projection.c translation.c methods.c compose.c unary_operators.c \
	utils.c interprocedural.c prettyprint.c
LIB_HEADERS= effects-convex-local.h
LIB_OBJECTS= interface.o old_utils.o old_prettyprint.o old_combine.o \
	old_projection.o translation.o methods.o compose.o unary_operators.o \
	utils.o interprocedural.o prettyprint.o




