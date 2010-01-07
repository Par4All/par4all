# $Id$
#
# Copyright 1989-2010 MINES ParisTech
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

#debug_output := $(shell echo libraries.mk  > /dev/tty)

######################################################################## NEWGEN

# Enable NewGen DOoM model to have XML backend and XPath functionalities.

ifdef USE_NEWGEN_DOOM

NEWGEN_DOOM_INCLUDES = $(shell pkg-config --cflags glib-2.0 libxml-2.0)
NEWGEN_DOOM_LIBS = $(shell pkg-config --libs glib-2.0 libxml-2.0)
CPPFLAGS += -DUSE_NEWGEN_DOOM

else

NEWGEN_DOOM_INCLUDES =
NEWGEN_DOOM_LIBS =

endif

newgen.libs	= genC $(NEWGEN_DOOM_LIBS)

VERSION=0.1

