#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#

LIB_CFILES	=	print.c print_code_as_a_graph.c cprettyprinter.c
LIB_HEADERS	=	prettyprint-local.h
LIB_OBJECTS	=	$(LIB_CFILES:.c=.o)
