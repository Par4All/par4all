#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/20 18:36:13 $, 

SOURCES	= properties-rc.tex

INSTALL_DOC =	properties-rc.ps
INSTALL_HTM =	properties-rc.html \
		properties-rc 
INSTALL_SHR = 	properties.rc

all: $(INSTALL_SHR) $(INSTALL_DOC) $(INSTALL_HTM)

properties.rc: properties-rc.tex
	#
	# building properties.rc
	#
	sed 's/^[\/ ]\*\/*/# /' $(PIPS_ROOT)/Include/auto.h > $@
	sed 's,	,    ,g;s/ *$$//;/^alias /d' $< | filter_verbatim >> $@

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_HTM) $(INSTALL_DOC) $(INSTALL_SHR) *.dvi

# end of $RCSfile: config.makefile,v $
#
