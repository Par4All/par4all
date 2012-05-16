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

CPPFLAGS += -Dsparc 
LDFLAGS	+= -fast 

include $(ROOT)/makes/DEFAULT.mk

CC	= acc -temp=$(pips_home)/tmp
CFLAGS	= -g -fast -Xc

# The SC3 acc compiler forget to define the sparc flag;
LD	= $(CC) -bsdmalloc

RANLIB	= granlib

# lex broken for properties...
LEX	= flex

FFLAGS	= -O -g -U -u -C 
