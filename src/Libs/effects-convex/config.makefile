#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#

# old_prettyprint.c old_utils.c 


LIB_CFILES =	interface.c \
		translation.c \
		methods.c \
		compose.c \
		unary_operators.c \
		utils.c \
		debug.c \
		interprocedural.c \
		prettyprint.c \
		old_combine.c \
		old_projection.c

LIB_HEADERS =	effects-convex-local.h

LIB_OBJECTS =	$(LIB_CFILES:.c=.o)
