# $Id$
#
# Copyright 1989-2014 MINES ParisTech
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

TARGET	= arithmetique

LIB_CFILES =	abs.c \
		divide.c \
		exp.c \
		modulo.c \
		pgcd.c \
		ppcm.c \
		io.c \
		errors.c \
		value.c

OTHER_HEADERS =	assert.h \
		boolean.h \
		arithmetic_errors.h \
		linear.h \
		linear_version.h

INC_TARGET	= $(TARGET).h

LIB_TARGET	= lib$(TARGET).a

INSTALL_INC	+= $(OTHER_HEADERS)

# regenerate everytime
.PHONY: linear_version.h
VERSION_SH = ../../makes/version.sh
linear_version.h:
	{ \
	  echo '#define LINEAR_REV "$(shell $(VERSION_SH) ../..)"' ; \
	  echo '#define LINEAR_MAKE_REV "$(shell $(VERSION_SH) ../../makes)"' ; \
	} > $@
 
clean: clean-version
clean-version:
	$(RM) linear_version.h

################################################################## COMMON STUFF

ROOT    = ../..
PROJECT = linear
include $(ROOT)/makes/main.mk
