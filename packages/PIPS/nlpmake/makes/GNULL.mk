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

include $(ROOT)/makes/DEFAULT.mk
include $(ROOT)/makes/gnu-stuff.mk
include $(ROOT)/makes/longlong.mk

# -ansi -petantic-errors and long long int is not a good idea.
CANSI=

# must force definition of long long int constants?
# CPPFLAGS+= -D__GNU_LIBRARY__ -D__USE_GNU
