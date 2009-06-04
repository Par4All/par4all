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

TARGET	= preprocessor

INC_CFILES = \
	initializer.c \
	source_file.c \
	split_file.c \
	csplit_file.c \
	module.c

DERIVED_HEADERS = splitc.h
DERIVED_CFILES = splitcyaccer.c splitclexer.c

LIB_CFILES =	$(DERIVED_CFILES) $(INC_CFILES)

INC_TARGET = $(TARGET).h
LIB_TARGET = lib$(TARGET).a

YOPT	= -d

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) y.tab.c y.tab.h

# split C parser

YY2SC	= sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/SPLITC_/g;s/yy/splitc_/g'

splitcyaccer.c: splitc.y
	$(PARSE) `pwd`/splitc.y
	$(YY2SC) y.tab.c > splitcyaccer.c
	$(YY2SC) y.tab.h > splitc.h
	$(RM) y.tab.c y.tab.h

splitc.h: splitcyaccer.c

splitclexer.c: splitc.l splitc.h
	$(SCAN) `pwd`/splitc.l | $(YY2SC) > $@
