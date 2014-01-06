#
# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of NewGen.
#
# NewGen is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License along with
# NewGen.  If not, see <http://www.gnu.org/licenses/>.
#

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

INSTALL_INC	= $(LIB_HEADERS) $(DERIVED_LIB_HEADERS) newgen_version.h

LIB_TARGET	= libgenC.a

BIN_TARGET	= newC

