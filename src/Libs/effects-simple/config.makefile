#
# $Id$
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#

LIB_CFILES	= \
	binary_operators.c \
	interface.c \
	methods.c \
	interprocedural.c \
	prettyprint.c \
	unary_operators.c \
	filter_proper_effects.c

LIB_HEADERS	= effects-simple-local.h

LIB_OBJECTS	= $(LIB_CFILES:%.c=%.o)

