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

TARGET=union
BUILT_SOURCES=$(TARGET).h sl_gram.h
nodist_include_HEADERS=$(TARGET).h
AM_CPPFLAGS=-I$(srcdir)/../arithmetique -I../arithmetique -I../vecteur -I../contrainte -I../sc -I../sommet -I../polyedre -I../ray_dte -I../sg -I../polynome -I$(top_srcdir)
AM_YFLAGS=-p slx_ -d
AM_LFLAGS=-p slx_

noinst_LTLIBRARIES=libunion.la
libunion_la_SOURCES=disjunct.c \
		path.c \
		reduc.c \
		sc_list.c\
		sl_gram.y \
		sl_lex.l

CLEANFILES=sl_gram.[ch] sl_lex.c

include $(top_srcdir)/makes/cproto.mk
