#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/20 15:59:05 $, 

SOURCES=	pipsmake-rc.tex \
		make-pips-menu \
		make-builder-map \
		auto.h

INSTALL_SHR=	pipsmake.rc \
		wpips.rc

INSTALL_INC=	resources.h \
		phases.h \
		builder_map.h \
		wpips_transform_menu_layout.h

INSTALL_DOC=	pipsmake-rc.dvi

DERIVED_FILES=	$(INSTALL_SHR) $(INSTALL_INC) $(INSTALL_DOC)

all: $(DERIVED_FILES)

pipsmake.rc: pipsmake-rc.tex
	#
	# building pipsmake.rc
	#
	sed 's,	,    ,g;s/ *$$//;/^alias /d' $< | filter_verbatim | \
		cat auto.h - > $@

wpips.rc: pipsmake-rc.tex
	#
	# buidling wpips.rc
	#
	sed 's/^[\/ ]\*\/*/--/' auto.h > $@
	sed 's,	,    ,g;s/ *$$//;/^alias /!d' $< >> $@

resources.h: pipsmake.rc
	#
	# building resources.h
	# 
	cpp $< | \
	sed '/>/!d;s/^.*MODULE\.//;s/^.*PROGRAM\.//;\
		s/^.*ALL\.//;s/^.*MAIN\.//;' | \
	tr '[a-z]' '[A-Z]' | sort -u | sed 's/.*/#define DBR_& "&"/' | \
	cat auto.h - > $@

phases.h: pipsmake.rc
	#
	# building phases.h
	#
	cpp $< | sed '/^[a-z]/!d;s/ .*//g;' | tr '[a-z]' '[A-Z]' | sort -u | \
	sed 's/.*/#define BUILDER_& "&"/' | cat auto.h - > $@

builder_map.h: pipsmake.rc
	#
	# building builder_map.h
	# 
	sh ./make-builder-map < $< > $@


wpips_transform_menu_layout.h:  pipsmake-rc.tex
	# 
	# building wpips_transform_menu_layout.h
	sh ./make-pips-menu < pipsmake-rc.tex

clean: local-clean

local-clean:
	$(RM) $(DERIVED_FILES) *.aux *.log *.ind *.idx

# end of $RCSfile: config.makefile,v $
#
