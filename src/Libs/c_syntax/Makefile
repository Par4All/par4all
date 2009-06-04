# $Id$
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

TARGET	= c_syntax
YOPT	=	-d

INC_CFILES =	c_parser.c \
		statement.c \
		util.c

DERIVED_HEADERS = cyacc.h
DERIVED_CFILES	= cyacc.c clexer.c

LIB_CFILES = 	$(INC_CFILES) $(DERIVED_CFILES)

INC_TARGET =    $(TARGET).h
LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) y.tab.[ch]

# C parser

YY2C	= sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/C_/g;s/yy/c_/g'

cyacc.c: cyacc.y
	$(PARSE) `pwd`/cyacc.y
	$(YY2C) y.tab.c > cyacc.c
	$(YY2C) y.tab.h > cyacc.h
	$(RM) y.tab.c y.tab.h

cyacc.h: cyacc.c

clexer.c: clex.l cyacc.h
	$(SCAN) `pwd`/clex.l | $(YY2C) > $@
