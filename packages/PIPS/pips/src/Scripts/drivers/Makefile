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
ROOT    = ../../..
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
	$(INSTALL) -d $(EXE.d)
	for l in $(LINKS) ; do $(RM) $(EXE.d)/$$l ; done
	for l in $(LINKS) ; do ln -s pips.sh $(EXE.d)/$$l ; done

unbuild:links-unbuild
links-unbuild:
	for l in $(LINKS) ; do $(RM) $(EXE.d)/$$l ; done
	test ! -d $(EXE.d) || $(RMDIR) --ignore-fail-on-non-empty $(EXE.d)

clean: links-clean

links-clean:
	$(RM) .build_links
