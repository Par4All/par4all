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

TARGET	= tpips
YOPT	= -v -d
INC_CFILES	= tpips.c wrapper.c revisions.c
OTHER_CFILES	= main_tpips.c
DERIVED_HEADERS	= tp_yacc.h completion_list.h
DERIVED_CFILES	= tp_yacc.c tp_lex.c
DERIVED_FILES	= y.output

LIB_CFILES	= $(INC_CFILES) $(DERIVED_CFILES)
INC_TARGET	= $(TARGET).h
LIB_TARGET	= lib$(TARGET).a
BIN_TARGET	= $(TARGET)

INSTALL_LIB = 	main_tpips.o

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) $(DERIVED_FILES)

$(ARCH)/tpips: $(ARCH)/$(TPIPS_MAIN)
$(ARCH)/$(LIB_TARGET): completion_list.h

# revision string...
include $(MAKE.d)/version.mk
$(ARCH)/revisions.o: pips_version.h
tpips.h: pips_version.h

# regenerate everytime
.PHONY: pips_version.h
VERSION_SH = $(MAKE.d)/version.sh
pips_version.h:
	{ \
	  echo '#define PIPS_REV "$(shell $(VERSION_SH) ../../..)"' ; \
	  echo '#define PIPS_MAKE_REV "$(shell $(VERSION_SH) ../../../makes)"' ; \
	  echo '#define CC_VERSION "$(shell $(CC_VERSION))"' ; \
	  echo '#define UTC_DATE "$(shell date -u)"' ; \
	} > $@

install-temporary-header: pips_version.h

# this may be missing when bootstrapping
$(ARCH)/tp_yacc.o: tpips.h

clean: clean-version
clean-version:
	$(RM) pips_version.h

# tpips parser
YY2TP	= sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/TP_/g;s/yy/tp_/g'

tp_lex.c: tp_lex.l tp_yacc.h
	$(SCAN) tp_lex.l | $(YY2TP) > tp_lex.c

tp_yacc.c: tp_yacc.y
	$(PARSE) tp_yacc.y
	$(YY2TP) y.tab.c > tp_yacc.c
	$(YY2TP) y.tab.h > tp_yacc.h
	$(RM) y.output y.tab.c y.tab.h

tp_yacc.h: tp_yacc.c

# tpips completion
completion_list.h :	$(INC.d)/resources.h \
			$(INC.d)/phases.h \
			$(INC.d)/printable_resources.h \
			$(ETC.d)/properties.rc
	./build_completion_lists.sh $(ROOT) > $@
