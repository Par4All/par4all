#
# The following macros define your library.
#
# Name of the library without the .a suffix.
TARGET= 	comp_sections
#

# Name of the main program to test or use the library
MAIN=		main
#

#		/usr/lib/debug/malloc.o 

# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	comp_sections.c propagate.c ss.c operators.c myintrinsics.c \
		dbase.c prettyprint.c
LIB_HEADERS=	comp_sections-local.h propagate.h ss.h  myintrinsics.h \
		base.h all.h
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)


