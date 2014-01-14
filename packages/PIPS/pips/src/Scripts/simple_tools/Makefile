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

# pips shell simple tools

SCRIPTS =	pips_c2openmp pips_f2openmp

INSTALL_EXE =	$(SCRIPTS)
INSTALL_SHR = simple_tools.bash

pips_c2openmp: pips_c2openmp.in
	sed -e 's,@datadir@,$${INSTALL_DIR:-$$PIPS_ROOT}/share,g' $< > $@
	chmod +x $@

pips_f2openmp: pips_f2openmp.in
	sed -e 's,@datadir@,$${INSTALL_DIR:-$$PIPS_ROOT}/share,g' $< > $@
	chmod +x $@

clean: local-clean
local-clean:
	$(RM) $(SCRIPTS)

# common stuff
ROOT    = ../../..
#PROJECT = pips
include $(ROOT)/makes/main.mk
