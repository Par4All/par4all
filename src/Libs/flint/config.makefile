#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	flint.c flint_walk.c \
		flint_check.c flint_utils.c \
		uninitialized_variables.c
LIB_HEADERS=	flint-local.h local.h
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)
