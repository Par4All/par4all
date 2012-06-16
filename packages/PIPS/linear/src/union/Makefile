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

TARGET	= union

CWEB_CFILES =	disjunct.c \
		path.c \
		reduc.c \
		sc_list.c

CWEB_OTHERS = 	sl_lex.l \
		sl_gram.y \
		union-local.h

DERIVED_CFILES = \
		sl_gram.c \
		sl_lex.c

DERIVED_HEADERS = sl_gram.h

INC_CFILES = $(CWEB_CFILES)
LIB_CFILES = $(CWEB_CFILES) $(DERIVED_CFILES)

LIB_TARGET = lib$(TARGET).a
INC_TARGET = $(TARGET).h

################################################################## COMMON STUFF

ROOT    = ../..
PROJECT = linear
include $(ROOT)/makes/main.mk

# ignore default rules
%.c: %.w; touch $@
%.h: %.w; touch $@
%.tex: %.w

# can only work if nuweb executable is available
# could it work with cweb-latex noweb funnelweb fweb?
NUWEB	= nuweb
code:;	$(NUWEB) -t UNION.w
doc:;	$(NUWEB) -f -o  UNION.w

YY2SL	= sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/SL_/g;s/yy/slx_/g;'

sl_lex.c: sl_lex.l sl_gram.h
	$(SCAN) $< | $(YY2SL) > sl_lex.c

sl_gram.c: sl_gram.y
	$(PARSE) -d $< 
	$(YY2SL) y.tab.c > sl_gram.c
	$(YY2SL) y.tab.h > sl_gram.h
	$(RM) y.tab.c y.tab.h

sl_gram.h: sl_gram.c

clean: local-clean

local-clean:; $(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)
# If we were sure to have nuweb...
#local-clean:; $(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) $(CWEB_CFILE)
