# $Id: Makefile 880 2005-12-26 08:22:48Z coelho $

# Moved to a separate file to ease documentation generation.

########################################################## STANDARD DEFINES

LIB_HEADERS=\
        genC.h \
        newgen_assert.h \
        newgen_types.h \
        newgen_hash.h \
        newgen_set.h \
        newgen_list.h \
        newgen_generic_mapping.h \
        newgen_generic_stack.h \
        newgen_generic_function.h \
        newgen_map.h \
        newgen_stack.h \
	newgen_array.h \
	newgen_string_buffer.h

DERIVED_LIB_HEADERS	= \
	newgen_auto_string.h

OTHER_HEADERS=\
        newgen_include.h \
        read.y \
        read.l \
        gram.y \
        token.l

DERIVED_HEADERS=\
	genread.h \
	genspec.h \
	$(DERIVED_LIB_HEADERS)

DERIVED_CFILES=\
	genread_yacc.c \
	genread_lex.c \
	genspec_yacc.c \
	genspec_lex.c

LIB_CFILES=\
        build.c \
        genC.c \
        genClib.c \
	tabulated.c \
        hash.c \
        set.c \
        list.c \
        stack.c \
	array.c \
	string.c \
	string_buffer.c \
	$(DERIVED_CFILES)

OTHER_CFILES=\
	new.c \
	genLisp.c \
	genSML.c

INSTALL_INC	= $(LIB_HEADERS) $(DERIVED_LIB_HEADERS)

LIB_TARGET	= libgenC.a

BIN_TARGET	= newC

