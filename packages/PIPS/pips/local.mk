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
# $Id$
# some local add-on for pips compilation

clean: local-clean
build: local-clean
compile: local-clean

# set ARCH macro
MAKE.d	= ./makes
include $(MAKE.d)/arch.mk

FINDsrc	= find ./src -name '.svn' -type d -prune -o

local-clean:
	# remove executables to know if the compilation failed
	$(RM) bin/$(ARCH)/* lib/$(ARCH)/lib*.a lib/$(ARCH)/*.o
	$(FINDsrc) -name .build_lib.$(ARCH) -print0 | xargs -0 rm -f
