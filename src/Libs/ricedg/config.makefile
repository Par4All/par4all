#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#

LIB_CFILES =	util.c contexts.c testdep_util.c \
		ricedg.c prettyprint.c quick_privatize.c

LIB_HEADERS =	ricedg-local.h local.h

LIB_OBJECTS =	$(LIB_CFILES:.c=.o)

