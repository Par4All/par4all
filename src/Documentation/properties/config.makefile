#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/09/02 10:47:07 $, 

SOURCES	= properties-rc.tex

INSTALL_DOC =	properties-rc.ps
INSTALL_HTM =	properties-rc
INSTALL_SHR = 	properties.rc

all: $(INSTALL_SHR) $(INSTALL_DOC) properties-rc.html

properties.rc: properties-rc.tex
	#
	# building properties.rc
	#
	{ cat $(PIPS_ROOT)/Include/auto-number.h ;
	  sed 's,	,    ,g;s/ *$$//;/^alias /d' $< | filter_verbatim ; \
	} > $@

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_HTM) $(INSTALL_DOC) $(INSTALL_SHR) *.dvi

# end of $RCSfile: config.makefile,v $
#
