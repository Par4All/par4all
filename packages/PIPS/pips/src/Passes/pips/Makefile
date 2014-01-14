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

# Source, header and object files used to build the target

TARGET	= pips

LIB_CFILES	= pips.c revisions.c
OTHER_CFILES	= main_pips.c

INC_TARGET =    $(TARGET).h
LIB_TARGET =    lib$(TARGET).a
BIN_TARGET =	$(TARGET)

INSTALL_LIB = 	main_pips.o

# common stuff
ROOT    = ../../..
PROJECT	= pips
include $(ROOT)/makes/main.mk

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

clean: clean-version
clean-version:
	$(RM) pips_version.h


# local stuff
$(ARCH)/pips: $(ARCH)/$(PIPS_MAIN)
