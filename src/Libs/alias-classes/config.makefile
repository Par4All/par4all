# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES =	alias_pairs.c \
		alias_lists.c \
		alias_classes.c \
		prettyprint.c
LIB_HEADERS =	alias-classes-local.h
LIB_OBJECTS =	$(LIB_CFILES:.c=.o)

