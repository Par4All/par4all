# $Id$
#
# Copyright 1989-2012 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

# macros related to gnu compilation.

AR	= gar
ARFLAGS	= rv

CC	= gcc
CANSI	= -ansi -pedantic-errors
CFLAGS	= -g -O2 -Wall -W -pipe -std=c99
# ??? -MG
CMKDEP	= -MM

LD	= $(CC)
RANLIB	= granlib

ifdef PIPS_F77
	FC	= $(PIPS_F77)
else
	FC	= f77
endif

FFLAGS	= -O2 -g -Wimplicit -pipe

LDFLAGS += -g

# putenv() => svid
# getwd()  => bsd
# getopt() => posix2

CPPFLAGS += \
	-D__USE_FIXED_PROTOTYPES__

LEX	= flex
LFLAGS	= 

LINT	= lint
LINTFLAGS= -habxu
