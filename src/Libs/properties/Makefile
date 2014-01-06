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

TARGET	= properties

DERIVED_CFILES = properties.c

LIB_CFILES	= $(DERIVED_CFILES)

INC_TARGET	= $(TARGET).h

LIB_TARGET	= lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

properties.h: properties.c

properties.c: properties.l
	$(SCAN) `pwd`/$< | \
	sed -e '/^FILE \*yyin/s/=[^,;]*//g;s/YY/PROP_/g;s/yy/prop_/g;s/\(void *\*prop_alloc\)/static \1/;s/\(void *\*prop_realloc\)/static \1/;' \
		 > $@

clean: local-clean
local-clean:
	$(RM) $(DERIVED_CFILES)
