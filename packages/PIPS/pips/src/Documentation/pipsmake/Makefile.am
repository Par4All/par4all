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

if WITH_DOC
doc_DATA=pipsmake-rc.pdf
endif
include $(top_srcdir)/makes/latex.mk

# Convert idraw/eps files.
pipsmake-rc.pdf:user-and-source-files.pdf

dist_noinst_SCRIPTS=make-pips-menu make-builder-map generate_all_menu_documentation pipspassdep.pl

nodist_sysconf_DATA=pipsmake.rc properties.rc pipsdep.rc wpips.rc properties-2009-10-31.rc 
dist_noinst_DATA=pipsmake.sed pipsmake-rc.tex properties-2009-10-31.rc.model resources.sed user-and-source-files.idraw

BUILT_SOURCES=resources.h printable_resources.h phases.h builder_map.h wpips_transform_menu_layout.h wpips_view_menu_layout.h
nodist_include_HEADERS=resources.h printable_resources.h phases.h builder_map.h wpips_transform_menu_layout.h wpips_view_menu_layout.h

pipsmake_flt	= $(SED) -f $(srcdir)/pipsmake.sed
# hmmm... this is generated locally to the virtual directory
pipsprop_flt	= $(SED) -f pipsprop.sed
UPPER	= tr '[a-z]' '[A-Z]'

pipsprop.sed:pipsmake.sed
	$(SED) 's/PipsMake/PipsProp/g' $(srcdir)/pipsmake.sed > $@

pipsmake.rc:pipsmake-rc.tex
	$(pipsmake_flt) $(srcdir)/pipsmake-rc.tex | $(SED) '/^alias /d' > $@

properties.rc: pipsmake-rc.tex pipsprop.sed
	$(pipsprop_flt) $(srcdir)/pipsmake-rc.tex > $@

properties-2009-10-31.rc: properties-2009-10-31.rc.model
	cp $(srcdir)/properties-2009-10-31.rc.model $@

pipsdep.rc: pipspassdep.pl pipsmake-rc.tex
	$(PERL) $(srcdir)/pipspassdep.pl $(srcdir)/pipsmake-rc.tex > $@

wpips.rc: pipsmake-rc.tex
	$(pipsmake_flt) $(srcdir)/pipsmake-rc.tex | $(SED) '/^alias /!d' > $@

resources.h: pipsmake.rc
	$(SED) -f $(srcdir)/resources.sed pipsmake.rc | \
	$(UPPER) | sort -u | $(SED) 's/.*/#define DBR_& "&"/' > $@

printable_resources.h: resources.h
	$(SED) '/define.*FILE/!d' resources.h > $@

phases.h: pipsmake.rc
	$(SED) '/^[a-z]/!d;s/ .*//g;' pipsmake.rc | tr '[a-z]' '[A-Z]' | sort -u | \
	$(SED) 's/.*/#define BUILDER_& "&"/' > $@

builder_map.h: pipsmake.rc $(srcdir)/make-builder-map
	$(srcdir)/make-builder-map < pipsmake.rc > $@


epips_transform_menu_layout.el wpips_transform_menu_layout.h epips_view_menu_layout.el wpips_view_menu_layout.h:  pipsmake-rc.tex generate_all_menu_documentation
	$(srcdir)/generate_all_menu_documentation -layout < $(srcdir)/pipsmake-rc.tex

CLEANFILES=$(nodist_include_HEADERS) $(sysconf_DATA) $(BUILT_SOURCES) pipsprop.sed epips_transform_menu_layout.el epips_view_menu_layout.el user-and-source-files.pdf $(nodist_sysconf_DATA)
