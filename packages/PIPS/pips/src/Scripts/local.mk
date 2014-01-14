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

clean: NO_INCLUDES=1
export NO_INCLUDES

# ??? hack
# We need to read the $(ARCH).mk to know if we need to compile jpips or not:
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../..
endif
MAKE.d	= $(ROOT)/makes
include $(MAKE.d)/root.mk
include $(MAKE.d)/arch.mk
include $(MAKE.d)/$(ARCH).mk

#debug_output := $(shell echo no_jpips.mk  > /dev/tty)

# not needed? stats stf jpips make
FWD_DIRS =	drivers dev env graph misc validation hpfc simple_tools step

# Compile epips only if needed:
ifndef PIPS_NO_EPIPS
	FWD_DIRS += epips
endif

# Compile Jpips only if needed:
ifndef PIPS_NO_JPIPS
	FWD_DIRS += jpips
endif

FWD_PARALLEL = 1
