#
# $Id$
# 
# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES	= \
	proper_effects_engine.c \
	rw_effects_engine.c \
	in_effects_engine.c \
	out_effects_engine.c \
	mappings.c \
	unary_operators.c \
	binary_operators.c \
	utils.c \
	prettyprint.c \
	intrinsics.c

# 	interprocedural.c

LIB_HEADERS	= effects-generic-local.h

LIB_OBJECTS	= $(LIB_CFILES:%.c=%.o)

