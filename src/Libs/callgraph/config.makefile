#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES =	callgraph.c print.c graph.c
LIB_HEADERS =	callgraph-local.h
LIB_OBJECTS =	$(LIB_CFILES:.c=.o)
