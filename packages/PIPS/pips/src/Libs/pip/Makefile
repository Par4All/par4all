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

TARGET	= pip

INC_CFILES =	pip.c \
		ps_max_to_min.c \
		ps_to_fic_pip.c \
		solpip.c \
		pip_interface.c \
		sol.c \
		traiter.c \
		tab.c \
		integrer.c \
		maind.c
INC_YFILES=solpip_parse.y solpip_scan.l

DERIVED_CFILES = \
		solpip_parse.c \
		solpip_scan.c

DERIVED_HEADERS	= solpip_parse.h

PIP_HEADERS =	pip__type.h \
		pip__tab.h \
		pip__sol.h

LIB_CFILES =	$(INC_CFILES) $(DERIVED_CFILES)

INSTALL_INC =	$(PIP_HEADERS)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)

$(TARGET).h: $(DERIVED_HEADERS)

solpip_parse.c solpip_parse.h : solpip_parse.y
	${PARSE} -d solpip_parse.y
	sed -e '/extern char \*malloc/d;s/YY/QUAYY/g;s/yy/quayy/g' \
		y.tab.c > solpip_parse.c
	rm y.tab.c
	sed -e 's/YY/QUAYY/g;s/yy/quayy/g' y.tab.h > solpip_parse.h
	rm y.tab.h

solpip_scan.c : solpip_scan.l
	${SCAN} solpip_scan.l |\
	sed '/^FILE* yyin/s/=[^,;]*//g;s/YY/QUAYY/g;s/yy/quayy/g;' \
		> solpip_scan.c

