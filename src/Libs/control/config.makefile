#
# $Id$
#
# Source, header and object files used to build th1e library.
# Do not include the main program source file.

LIB_CFILES = \
	clean_up_sequences.c \
	control.c \
	graph.c \
	hierarchize.c \
	module.c \
	reorder.c \
	unspaghettify.c \
	cfg.c \
	unreachable.c \
	typing.c

LIB_HEADERS =	control-local.h

LIB_OBJECTS =	$(LIB_CFILES:.c=.o)
