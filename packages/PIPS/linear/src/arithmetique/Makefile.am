# $Id$
#
# Copyright 1989-2012 MINES ParisTech
#
# This file is part of Linear/C3 Library.
#
# Linear/C3 Library is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.
#

TARGET=arithmetique
noinst_LTLIBRARIES=libarithmetique.la

BUILT_SOURCES=$(TARGET).h 
nodist_include_HEADERS=$(TARGET).h
dist_include_HEADERS =	assert.h \
		boolean.h \
		arithmetic_errors.h \
		linear.h

libarithmetique_la_SOURCES=abs.c \
		divide.c \
		exp.c \
		modulo.c \
		pgcd.c \
		ppcm.c \
		io.c \
		errors.c \
		value.c

include $(top_srcdir)/makes/cproto.mk
