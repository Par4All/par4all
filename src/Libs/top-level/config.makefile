#
# $Id$
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#

LIB_CFILES =	dbm.c \
		help.c \
		newgen.c \
		logging.c \
		view.c 

LIB_HEADERS =	top-level-local.h

LIB_OBJECTS =	$(LIB_CFILES:.c=.o) 
