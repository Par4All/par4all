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

F.sh = 	hpfc \
	hpfc_directives \
	hpfc_compile \
	hpfc_delete \
	hpfc_install

BIN_TARGET = hpfc_interactive

INSTALL_EXE = $(F.sh)

INSTALL_SHR = hpfc_stubs.f hpfc_stubs.direct.f

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(ARCH)/hpfc_interactive.o: $(ARCH)/.dir

$(ARCH)/hpfc_interactive: $(ARCH)/hpfc_interactive.o
	$(LINK) $(ARCH)/hpfc_interactive \
		$(ARCH)/hpfc_interactive.o -lreadline -lncurses

# the direct version of the stubs need not be filtered by hpfc_directives.
hpfc_stubs.direct.f: hpfc_stubs.f
	# building $@ from $<
	sed 's,^!fcd\$$ fake,      call hpfc9,' $< > $@

clean: local-clean

local-clean:
	$(RM) hpfc_stubs.direct.f
