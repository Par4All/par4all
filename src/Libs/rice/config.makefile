#
# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES =	rice.c codegen.c scc.c

LIB_HEADERS =	rice-local.h local.h

LIB_OBJECTS =	$(LIB_CFILES:.c=.o)
