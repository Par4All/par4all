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

TARGET	= sac

INC_CFILES = \
	simd_loop_const_elim.c if_conversion_compact.c if_conversion_init.c simd_loop_unroll.c simdizer.c varwidth.c codegen.c unroll.c operatorid.c \
	treematch.c atomizer.c vectransform.c reductions.c \
	singleass.c deatomizer.c if_conversion.c simd_memory_packing.c

DERIVED_CFILES = patterns.tab.c patterns.lex.c

DERIVED_HEADERS = patterns.tab.h

LIB_CFILES = $(INC_CFILES) $(DERIVED_CFILES)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

INSTALL_ETC = patterns.def

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_HEADERS) $(DERIVED_CFILES) 

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

patterns.lex.c: patterns.l patterns.tab.h
	flex -Cf -opatterns.lex.c -Ppatterns_yy patterns.l

patterns.tab.c patterns.tab.h: patterns.y
	bison -o patterns.tab.c -d -p patterns_yy patterns.y
