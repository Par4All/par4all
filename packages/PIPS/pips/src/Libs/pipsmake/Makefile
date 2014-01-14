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

TARGET	= pipsmake
YOPT	= -d
INC_CFILES =	pipsmake.c \
		activate.c \
		openclose.c \
		callback.c \
		unsplit.c \
		stubs.c
INC_YFILES=readmakefile.y lexer.l

DERIVED_HEADERS = readmakefile.h
DERIVED_CFILES = readmakefile.c lexer.c

LIB_CFILES =	$(DERIVED_CFILES)  $(INC_CFILES)
INC_TARGET = $(TARGET).h
LIB_TARGET = lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local specials
clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)

$(TARGET).h: $(DERIVED_CFILES)

# pipsmake parser

YY2PM	= sed -e '/^FILE \*yyin/s/=[^,;]*//g;s/YY/PIPSMAKE_/g;s/yy/pipsmake_/g'

lexer.c: lexer.l
	$(SCAN) lexer.l | $(YY2PM) > $@

readmakefile.c: readmakefile.y
	$(PARSE) readmakefile.y
	$(YY2PM) y.tab.c > readmakefile.c
	$(YY2PM) y.tab.h > readmakefile.h
	$(RM) y.tab.[hc] y.output

readmakefile.h: readmakefile.c
