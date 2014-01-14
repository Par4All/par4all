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

# configuration for FPIPS.

TARGET	= fpips

LIB_CFILES	= fpips.c

OTHER_CFILES	= main_fpips.c

CPPOPT =	$(FPIPS_CPPFLAGS)

LIB_TARGET	= lib$(TARGET).a

BIN_TARGET	= $(TARGET)

INSTALL_LIB = 	main_fpips.o

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# fix fpips inclusions
ifdef PIPS_NO_GPIPS
CPPFLAGS        +=      -DFPIPS_WITHOUT_GPIPS
endif # PIPS_NO_GPIPS

ifdef PIPS_NO_WPIPS
CPPFLAGS 	+=	-DFPIPS_WITHOUT_WPIPS
endif # PIPS_NO_WPIPS

# local stuff
$(ARCH)/fpips: $(ARCH)/$(FPIPS_MAIN)

clean: local-clean

local-clean:
	$(RM) $(ARCH)/pips $(ARCH)/tpips $(ARCH)/wpips $(ARCH)/fpips

ln: local-clean
	#
	# links
	#
	ln $(ARCH)/fpips $(ARCH)/gpips
	ln $(ARCH)/fpips $(ARCH)/pips
	ln $(ARCH)/fpips $(ARCH)/tpips
	ln $(ARCH)/fpips $(ARCH)/wpips
