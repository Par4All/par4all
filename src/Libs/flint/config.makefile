#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	flint.c flint_walk.c flint_check.c flint_utils.c uninitialized_variables.c
LIB_HEADERS=	flint-local.h
LIB_OBJECTS=	flint.o flint_walk.o flint_check.o flint_utils.o uninitialized_variables.o
