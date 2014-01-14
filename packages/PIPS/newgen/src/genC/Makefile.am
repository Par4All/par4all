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
SUBDIRS=genread genspec

lib_LTLIBRARIES=libnewgenlibs.la
bin_PROGRAMS=newC

newC_SOURCES=new.c
newC_LDADD=libnewgenlibs.la

include_HEADERS=\
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
		newgen_string_buffer.h \
		newgen_include.h

nodist_include_HEADERS=newgen_auto_string.h

libnewgenlibs_la_SOURCES=\
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
		string_buffer.c

libnewgenlibs_la_LDFLAGS=$(LDFLAGS_NO_UNDEFINED)
libnewgenlibs_la_LIBADD=genread/libgenread.la genspec/libgenspec.la ../gnulib/libgnu.la

BUILT_SOURCES=newgen_auto_string.h
AM_CPPFLAGS=-Igenread -I$(srcdir)/genread -Igenspec -I$(srcdir)/genspec

newgen_auto_string.h:string.c
	$(CPROTO) -evcf2 -E "$(CPP) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" \
		`(test -f string.c && echo string.c ) || echo $(srcdir)/string.c` > newgen_auto_string.h

CLEANFILES=$(nodist_include_HEADERS)

