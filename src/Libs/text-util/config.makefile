#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#
LIB_CFILES=	text_print.c util.c
LIB_HEADERS=	text-util-local.h
LIB_OBJECTS=	$(LIB_OBJECTS:.c=.o)
