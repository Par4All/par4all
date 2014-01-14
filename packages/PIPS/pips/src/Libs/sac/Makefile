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

TARGET	= sac

INC_CFILES = \
	if_conversion_compact.c if_conversion_init.c simdizer.c varwidth.c codegen.c unroll.c operatorid.c \
	treematch.c atomizer.c vectransform.c reductions.c \
	singleass.c deatomizer.c if_conversion.c simd_memory_packing.c

INC_YFILES = patterns.y lexer.l

DERIVED_CFILES = patterns.c lexer.c

DERIVED_HEADERS = patterns.h

LIB_CFILES = $(INC_CFILES) $(DERIVED_CFILES)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

INSTALL_ETC = patterns.def

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_HEADERS) $(DERIVED_CFILES) 

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

lexer.c: lexer.l patterns.h
	sed -e 's/yy/patterns_yy/g' lexer.l > tmp.l
	flex -Cf -olexer.c -Ppatterns_yy tmp.l
	rm tmp.l
lexer.h:lexer.c

patterns.c : patterns.y
	sed -e 's/yy/patterns_yy/g' patterns.y > tmp.y
	bison -o patterns.c -d -p patterns_yy tmp.y
	rm tmp.y

patterns.h: patterns.c
