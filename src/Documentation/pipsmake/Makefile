#
# $Id$
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

AUTO =	$(PIPS_ROOT)/Include/auto

all: $(DERIVED_FILES)

# the documentation must generate its derived headers as soon as possible
# beacause it is expected to compute the dependencies of some librairies
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
	sed '/^[a-z]/!d;s/ .*//g;' $< | tr '[a-z]' '[A-Z]' | sort -u | \
	sed 's/.*/#define BUILDER_& "&"/' > $@

builder_map.h: pipsmake.rc
	#
	# building builder_map.h
	#
	./make-builder-map < $< > $@


epips_transform_menu_layout.el wpips_transform_menu_layout.h epips_view_menu_layout.el wpips_view_menu_layout.h:  pipsmake-rc.tex
	#
	# building menu layout files:
	#
	./generate_all_menu_documentation -layout < $<

headers: $(DERIVED_INC)

pipsmake-rc.pdf: user-and-source-files.pdf

clean: local-clean

local-clean:
	$(RM) -r $(DERIVED_FILES) pipsmake-rc.htdoc pipsprop.sed \
		user-and-source-files.pdf \
		*.aux *.log *.ind *.idx *.toc *.ilg *.dvi *.ps *.out *.brf

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk
