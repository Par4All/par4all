#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	
LIB_HEADERS=	properties-local.h properties.l
LIB_OBJECTS=	properties.o

DERIVED_CFILES = properties.c
INC_CFILES= $(DERIVED_CFILES)

default: all

properties.c: properties.l
	$(SCAN) properties.l | \
		sed -e 's/YY/PROP_/g;s/yy/prop_/g' > properties.c

depend: $(DERIVED_CFILES)
