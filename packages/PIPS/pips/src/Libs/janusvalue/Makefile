# $Id: Makefile 1432 2011-12-31 08:19:07Z coelho $
#
# Copyright 1989-2010 MINES ParisTech
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

TARGET	= janusvalue

LIB_CFILES =	isolve.c \
		r1.c

OTHER_HEADERS = \
		iabrev.h \
		iproblem.h \
		rproblem.h

LIB_TARGET	= lib$(TARGET).a

INC_TARGET	= $(TARGET).h

INSTALL_INC	+= $(OTHER_HEADERS)

################################################################## COMMON STUFF

ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
