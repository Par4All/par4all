# $Id$
#
# Copyright 1989-2014 MINES ParisTech
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

TARGET	= newgen

INC_CFILES = \
		newgen.c \
		Pvecteur.c \
		Ppolynome.c \
		Psc.c

INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

ROOT    = ../../..

# where to find newgen generated files...
inc_dir = $(ROOT)/include

NEWGEN_CFILES = $(notdir $(wildcard $(inc_dir)/*.c))

LIB_CFILES =	$(INC_CFILES) $(NEWGEN_CFILES)

CPPOPT	= -I$(inc_dir)

%.c: 
	# $(inc_dir)/$@
	ln -s $(inc_dir)/$@

PROJECT = pips
include $(ROOT)/makes/main.mk

clean: local-clean

local-clean:; $(RM) $(NEWGEN_CFILES)
