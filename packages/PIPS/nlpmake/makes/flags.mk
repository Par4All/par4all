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

# macros related to any compilation.
# put $(ROOT) ahead so as to override *_ROOT

CPPFLAGS += \
	-I. \
	-I$(ROOT)/include \
	-I$(NEWGEN_ROOT)/include \
	-I$(LINEAR_ROOT)/include \
	-I$(PIPS_ROOT)/include \
	-I$(EXTERN_ROOT)/include 

LDFLAGS += \
	-L./$(ARCH) \
	-L$(ROOT)/lib/$(ARCH) \
	-L$(PIPS_ROOT)/lib/$(ARCH) \
	-L$(NEWGEN_ROOT)/lib/$(ARCH) \
	-L$(LINEAR_ROOT)/lib/$(ARCH) \
	-L$(EXTERN_ROOT)/lib/$(ARCH) \
	-L$(LIB.d)/$(ARCH)
