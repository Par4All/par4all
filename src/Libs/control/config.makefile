#	%A% ($Date: 1997/08/01 14:42:09 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	


# Source, header and object files used to build th1e library.
# Do not include the main program source file.
LIB_CFILES=	clean_up_sequences.c control.c graph.c hierarchize.c module.c \
	reorder.c unspaghettify.c
LIB_HEADERS=	control-local.h
LIB_OBJECTS=	clean_up_sequences.o control.o graph.o hierarchize.o module.o \
	reorder.o unspaghettify.o
