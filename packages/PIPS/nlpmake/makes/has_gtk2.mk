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

ifndef has_gtk2_done

# first check that pkg-config is available? (comes with gnome?)
has_pkgcfg := $(shell type pkg-config > /dev/null 2>&1 && echo ok)

ifeq ($(has_pkgcfg),ok)

has_gtk2   := $(shell pkg-config --exists gtk+-2.0 && echo ok)

ifneq ($(has_gtk2),ok)

# no pkg-config => no gpips

$(warning "skipping gpips compilation, gtk2 is not available")
PIPS_NO_GPIPS	= 1

endif # has_gtk2

else # has_pkgcfg not ok

$(warning "skipping gpips compilation, pkg-config not found")
PIPS_NO_GPIPS	= 1

endif # has_pkgcfg

has_gtk2_done	= 1

endif # has_gtk2_done
