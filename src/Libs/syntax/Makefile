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

TARGET	= syntax

# I wanna the header file for the lexer
YOPT	=	-d

INC_CFILES= 	declaration.c \
		expression.c \
		equivalence.c \
		parser.c \
		procedure.c \
		reader.c \
		statement.c \
		return.c \
		malloc-info.c \
		macros.c
INC_YFILES= gram.y scanner.l

DERIVED_HEADERS = keywtbl.h tokyacc.h syn_yacc.h yacc.in
DERIVED_CFILES = syn_yacc.c scanner.c

LIB_CFILES	= $(INC_CFILES) $(DERIVED_CFILES)

INC_TARGET =    $(TARGET).h
LIB_TARGET =    lib$(TARGET).a

VALIDATE_TARGET	= Syntax

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) yacc.in

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

YY2SYN	= sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/SYN_/g;s/yy/syn_/g'

yacc.in: tokyacc.h gram.y
	cat tokyacc.h gram.y > $@

syn_yacc.c: yacc.in
	$(PARSE) $<
	$(YY2SYN) y.tab.c > syn_yacc.c
	$(YY2SYN) y.tab.h > syn_yacc.h
	$(RM) y.tab.c y.tab.h

syn_yacc.h: syn_yacc.c

scanner.c: scanner.l syn_yacc.h
	$(SCAN) scanner.l | $(YY2SYN) > $@

keywtbl.h: warning.h f77keywords
	{ cat warning.h ; \
	  echo "#include \"syn_yacc.h\"" ; \
	  echo "static struct Skeyword keywtbl[] = {" ;\
	  sed "s/^.*/{\"&\", TK_&},/" f77keywords ;\
	  echo "{0, 0}" ;\
	  echo "};" ; } > $@

tokyacc.h: warning.h f77keywords f77symboles
	{ cat warning.h ; \
	sed 's,\([^A-Z]*\)\(.*\),%token \1 TK_\2,' f77keywords f77symboles ;\
	} > $@
