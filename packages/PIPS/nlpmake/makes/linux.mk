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

# updates to GNU for LINUX.

# CFLAGS	+=	-march=pentium -mwide-multiply -malign-double

CPPFLAGS += \
	-D_POSIX_SOURCE \
	-D_POSIX_C_SOURCE=2 \
	-D_BSD_SOURCE \
	-D_SVID_SOURCE \
	-D_GNU_SOURCE

AR	=	ar
RANLIB	=	ranlib
DIFF	=	diff
M4	=	m4

# end of it
#
