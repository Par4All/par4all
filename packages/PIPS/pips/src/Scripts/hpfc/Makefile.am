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

dist_bin_SCRIPTS = 	hpfc \
	hpfc_directives \
	hpfc_compile \
	hpfc_delete \
	hpfc_install

bin_PROGRAMS= hpfc_interactive
hpfc_interactive_SOURCES=hpfc_interactive.c
AM_CPPFLAGS=$(READLINE_CPPFLAGS) $(NCURSES_CPPFLAGS)
hpfc_interactive_LDFLAGS=$(READLINE_LIBS) $(NCURSES_LIBS)

dist_data_DATA= hpfc_stubs.f hpfc_stubs.direct.f


# the direct version of the stubs need not be filtered by hpfc_directives.
hpfc_stubs.direct.f: hpfc_stubs.f
	$(SED) 's,^!fcd\$$ fake,      call hpfc9,' $(srcdir)/hpfc_stubs.f > hpfc_stubs.direct.f

CLEANFILES=hpfc_stubs.direct.f
