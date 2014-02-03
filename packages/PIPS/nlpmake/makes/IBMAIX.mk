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

include $(ROOT)/makes/DEFAULT.mk

# ansi required for newgen (otherwise __STDC__ or __STRICT_ANSI__ not def).
CANSI	= -qlanglvl=ansi 

# -ma to allow alloca():
CFLAGS	= -g -O2 -qmaxmem=8192 -qfullpath -ma 

# if -E is ommitted, the file is *compiled*:-(
CMKDEP	= -E -M

# -lbsd added so signal work:
LDFLAGS	+= -lbsd

FFLAGS	= -g -O2 -u

#
# others

TAR	= tar
ZIP	= gzip -v9
DIFF	= diff

#
# Well, AIX is not unix, and xlc does not have the expected behavior under -M
DEPFILE = *.u

# no -ltermcap under AIX. -lcurses instead.
TPIPS_ADDED_LIBS =	-lreadline -lcurses

# wpips was never compiled under aix.
include $(ROOT)/makes/no_wpips.mk

