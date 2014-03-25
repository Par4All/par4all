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

use_pdflatex	= 1

SOURCES=	pipsmake-rc.tex \
		make-pips-menu \
		make-builder-map \
		generate_all_menu_documentation \
		user-and-source-files.idraw

INSTALL_ETC=	epips_view_menu_layout.el \
		epips_transform_menu_layout.el \
		pipsmake.rc \
		properties.rc \
		properties-2009-10-31.rc \
		pipsdep.rc \
		wpips.rc

DERIVED_INC=	resources.h \
		printable_resources.h \
		phases.h \
		builder_map.h \
		wpips_transform_menu_layout.h \
		wpips_view_menu_layout.h

INSTALL_INC=	$(DERIVED_INC)
INSTALL_DOC=	pipsmake-rc.pdf
INSTALL_HTM=	pipsmake-rc.htdoc
# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

DERIVED_FILES=	$(INSTALL_ETC) $(DERIVED_INC) $(INSTALL_DOC)

all: $(DERIVED_FILES)

# the documentation must generate its derived headers as soon as possible
# because it is expected to compute the dependencies of some librairies
# especially Passes/tpips
phase0: .build_inc .build_etc

# skip latex 2 html if needed:
simple: $(DERIVED_INC) $(INSTALL_ETC) $(INSTALL_DOC)
simpleinstall: simple; $(MAKE) INSTALL_HTM= install

pipsmake.flt	= sed -f pipsmake.sed
pipsprop.flt	= sed -f pipsprop.sed

pipsprop.sed:	pipsmake.sed
	sed 's/PipsMake/PipsProp/g' $< > $@

pipsmake.rc: pipsmake-rc.tex
	#
	# building pipsmake.rc
	#
	$(pipsmake.flt) $< | sed '/^alias /d' > $@

properties.rc: pipsmake-rc.tex pipsprop.sed
	#
	# building properties.rc
	#
	$(pipsprop.flt) $< > $@

properties-2009-10-31.rc: properties-2009-10-31.rc.model
	#
	# building properties-2009-10-31.rc
	#
	cp properties-2009-10-31.rc.model properties-2009-10-31.rc

pipsdep.rc: pipspassdep.pl pipsmake-rc.tex
	#
	# building pipsdep.rc
	#
	./$^ > $@

wpips.rc: pipsmake-rc.tex
	#
	# buidling $@
	#
	$(pipsmake.flt) $< | sed '/^alias /!d' > $@

resources.h: pipsmake.rc
	#
	# building $@
	#
	sed -f resources.sed $< | \
	$(UPPER) | sort -u | sed 's/.*/#define DBR_& "&"/' > $@

printable_resources.h: resources.h
	#
	# building $@
	#
	sed '/define.*FILE/!d' $< > $@

phases.h: pipsmake.rc
	#
	# building $@
	#
	# The -u option for sort hides redeclared phases
	sed '/^[a-z]/!d;s/ .*//g;' $< | tr '[a-z]' '[A-Z]' | sort -u | \
	sed 's/.*/#define BUILDER_& "&"/' > $@

builder_map.h: pipsmake.rc
	#
	# building builder_map.h
	#
	./make-builder-map < $< > $@


epips_transform_menu_layout.el wpips_transform_menu_layout.h epips_view_menu_layout.el wpips_view_menu_layout.h:  pipsmake-rc.tex generate_all_menu_documentation
	#
	# building menu layout files:
	#
	./generate_all_menu_documentation -layout < $<

headers: $(DERIVED_INC)

pipsmake-rc.pdf: user-and-source-files.pdf
pipsmake-rc.dvi: user-and-source-files.eps

clean: local-clean

check: pispmake-rc.tex
	@echo "missing properties label:"
	pdflatex pipsmake-rc.tex 2>&1 | grep 'LaTeX Warning: Reference' | grep -E -o "prop:[^']+" | cut -d ':' -f 2


local-clean:
	$(RM) -r $(DERIVED_FILES) pipsmake-rc.htdoc pipsprop.sed \
		user-and-source-files.pdf user-and-source-files.eps \
		*.aux *.log *.ind *.idx *.toc *.ilg *.dvi *.ps *.out *.brf

# common stuff
ROOT    = ../../..
include $(ROOT)/makes/main.mk
