#	%A% ($Date: 1997/02/03 22:36:15 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	


# Source, header and object files used to build th1e library.
# Do not include the main program source file.
LIB_CFILES=	clean_up_sequences.c control.c module.c graph.c \
	unspaghettify.c reorder.c
LIB_HEADERS=	control-local.h
LIB_OBJECTS=	clean_up_sequences.o control.o module.o graph.o \
	unspaghettify.o reorder.o
