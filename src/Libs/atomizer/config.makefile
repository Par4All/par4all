#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES =	new_atomizer.c \
		atomizer.c \
		utils.c \
		codegen.c \
		control.c \
		norm_exp.c \
		defs_elim.c

LIB_HEADERS=	atomizer-local.h local.h

LIB_OBJECTS=	$(LIB_CFILES:.c=.o)

#
