# $Id$
#
# Copyright 1989-2014 MINES ParisTech
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

AR	= ar
ARFLAGS	= rv
CC	= cc
CFLAGS	= -O -g
CMKDEP	= -M
LD	= $(CC)
RANLIB	= ranlib
LEX	= flex
# Some parts really need flex:
FLEX	= flex
LFLAGS	=
FC	= f77
FFLAGS	= -O -g
LINT	= lint
LINTFLAGS= -habxu

# may need to be overwritten
CC_VERSION	= $(CC) --version | head -1

# The parser can no longer be compiled with yacc...
YACC	= bison
# Some parts really need bison:
BISON	= bison
YFLAGS	= -y

PROTO   = cproto
# do not use gcc -E here : it fails on .y and .l
PROTO_CPP = cpp
PRFLAGS    = -qevcf2

# A dummy target for the flymake-mode in Emacs:
check-syntax:
	$(COMPILE) -o nul.o -S ${CHK_SOURCES}

# end of it!
#
