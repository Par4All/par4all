#
# CFLAGS=		-g -ansi -Wall -mv8 -pipe
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=  binary_operators.c  interface.c methods.c interprocedural.c \
	prettyprint.c unary_operators.c
LIB_HEADERS= effects-simple-local.h
LIB_OBJECTS= binary_operators.o  interface.o methods.o interprocedural.o \
	prettyprint.o unary_operators.o



