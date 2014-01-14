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


#debug_output := $(shell echo local.mk  > /dev/tty)

clean: NO_INCLUDES=1
export NO_INCLUDES

# ??? hack
# We need to read the $(ARCH).mk to know if we need to compile wpips or not:
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../..
endif

MAKE.d	= $(ROOT)/makes
include $(MAKE.d)/root.mk
include $(MAKE.d)/arch.mk
include $(MAKE.d)/$(ARCH).mk

# check for gtk2 here, to know whether to forward to gpips or not
-include $(MAKE.d)/config.mk
include $(MAKE.d)/has_gtk2.mk
include $(MAKE.d)/has_swig.mk

FWD_DIRS	= pips tpips

# Skip compiling WPips if not required:

ifndef PIPS_NO_WPIPS
	FWD_DIRS	+= wpips
endif

ifndef PIPS_NO_GPIPS
	FWD_DIRS	+= gpips
endif

ifdef PIPS_ENABLE_PYPS
	FWD_DIRS	+= pypips
endif


ifdef PIPS_ENABLE_FORTRAN95
        FWD_DIRS        += fortran95
endif

# after its dependencies
FWD_DIRS	+= fpips

FWD_PARALLEL	= 1

fwd-fpips: fwd-pips fwd-tpips
