# $Id: Makefile.am 1434 2011-12-31 08:47:14Z coelho $
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
TARGET=janusvalue
BUILT_SOURCES=$(TARGET).h
EXTRA_DIST=$(TARGET)-local.h
AM_CPPFLAGS=-I$(srcdir)/../arithmetique -I../arithmetique

noinst_LTLIBRARIES=libjanusvalue.la
other_headers = \
		iabrev.h \
		iproblem.h \
		rproblem.h

libjanusvalue_la_SOURCES=isolve.c r1.c $(other_headers)

include $(top_srcdir)/makes/cproto.mk
