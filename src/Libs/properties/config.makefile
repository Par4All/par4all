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
	sed -e '/^FILE \*yyin/s/=[^,;]*//g;s/YY/PROP_/g;s/yy/prop_/g;s/\(void *\*prop_alloc\)/static \1/;s/\(void *\*prop_realloc\)/static \1/;' \
		 > properties.c

depend: $(DERIVED_CFILES)
