#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	properties.c
LIB_HEADERS=	properties-local.h properties.l
LIB_OBJECTS=	properties.o

default: all

properties.c: properties.l
	$(SCAN) properties.l | sed -e 's/YY/PP/g;s/yy/pp/g' > properties.c

depend: properties.c
