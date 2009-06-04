# $Id$
#
# Copyright 1989-2009 MINES ParisTech
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

# pips shell drivers

SCRIPTS =	Display \
		Pips \
		Init \
		Perform \
		Delete \
		Build \
		Select \
		pips.sh

INSTALL_EXE =	$(SCRIPTS)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

#PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff:
LINKS =	pips tpips wpips fpips gpips

.build_exe: .build_links

.build_links:
	$(MAKE) links
	touch $@

links:
	# no direct dependency, as usual
	$(MAKE) $(EXE.d)
	for l in $(LINKS) ; do $(RM) $(EXE.d)/$$l ; done
	for l in $(LINKS) ; do ln -s pips.sh $(EXE.d)/$$l ; done

clean: links-clean

links-clean:
	$(RM) .build_links
