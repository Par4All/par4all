# $Id: Makefile 1282 2009-08-11 10:16:01Z coelho $
#
# Copyright 1989-2009 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

TARGET=arithmetique
noinst_LTLIBRARIES=libarithmetique.la


other_headers =	assert.h \
		boolean.h \
		arithmetic_errors.h \
		linear.h \
		linear_version.h

BUILT_SOURCES=$(TARGET).h
nodist_include_HEADERS=$(TARGET).h $(other_headers)

libarithmetique_la_SOURCES=abs.c \
		divide.c \
		exp.c \
		modulo.c \
		pgcd.c \
		ppcm.c \
		io.c \
		errors.c \
		$(other_headers)

# work around a very "dumb" make
here   =  `[ "$(VPATH)" ] && echo $(VPATH) || echo '.'`

# regenerate everytime
.PHONY: linear_version.h
VERSION_SH = $(here)/../../makes/version.sh
linear_version.h:
	{ \
	  echo '#define LINEAR_REV "$(shell $(VERSION_SH) $(here)/../..)"' ; \
	  echo '#define LINEAR_MAKE_REV "$(shell $(VERSION_SH) $(here)/../../makes)"' ; \
	} > $@

CLEANFILES	= linear_version.h

include $(top_srcdir)/makes/cproto.mk
