#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/09/02 09:54:34 $, 

SOURCES=	pipsmake-rc.tex \
		make-pips-menu \
		make-builder-map

INSTALL_SHR=	pipsmake.rc \
		wpips.rc

DERIVED_INC=	resources.h \
		phases.h \
		builder_map.h \
		wpips_transform_menu_layout.h

INSTALL_INC=	$(DERIVED_INC)
INSTALL_DOC=	pipsmake-rc.ps
INSTALL_HTM=	pipsmake-rc 

DERIVED_FILES=	$(INSTALL_SHR) $(DERIVED_INC) $(INSTALL_DOC) pipsmake-rc.html

AUTO =	$(PIPS_ROOT)/Include/auto

all: $(DERIVED_FILES)

pipsmake.rc: pipsmake-rc.tex
	#
	# building pipsmake.rc
	#
	sed 's,	,    ,g;s/ *$$//;/^alias /d' $< | filter_verbatim | \
		cat $(AUTO).h - > $@

wpips.rc: pipsmake-rc.tex
	#
	# buidling wpips.rc
	#
	{ cat $(AUTO)-dash.h ;\
	  sed 's,	,    ,g;s/ *$$//;/^alias /!d' $< ; } > $@

resources.h: pipsmake.rc
	#
	# building resources.h
	# 
	cpp $< | \
	sed '/>/!d;s/^.*MODULE\.//;s/^.*PROGRAM\.//;\
		s/^.*ALL\.//;s/^.*MAIN\.//;' | \
	tr '[a-z]' '[A-Z]' | sort -u | sed 's/.*/#define DBR_& "&"/' | \
	cat $(AUTO).h - > $@

phases.h: pipsmake.rc
	#
	# building phases.h
	#
	cpp $< | sed '/^[a-z]/!d;s/ .*//g;' | tr '[a-z]' '[A-Z]' | sort -u | \
	sed 's/.*/#define BUILDER_& "&"/' | cat $(AUTO).h - > $@

builder_map.h: pipsmake.rc
	#
	# building builder_map.h
	# 
	sh ./make-builder-map < $< > $@


wpips_transform_menu_layout.h:  pipsmake-rc.tex
	# 
	# building wpips_transform_menu_layout.h
	#
	sh ./make-pips-menu < $< > $@

clean: local-clean

local-clean:
	$(RM) -r $(DERIVED_FILES) *.aux *.log *.ind *.idx *.toc *.ilg *.dvi

# end of $RCSfile: config.makefile,v $
#
