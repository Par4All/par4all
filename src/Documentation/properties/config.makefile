#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/20 18:27:40 $, 

SOURCES	= properties-rc.tex

INSTALL_DOC =	properties-rc.ps
INSTALL_HTM =	properties-rc.html \
		properties-rc 
INSTALL_SHT = 	properties.rc

properties.rc: properties-rc.tex
	#
	# building properties.rc
	#
	sed 's/^[\/ ]\*\/*/# /' $(PIPS_ROOT)/Include/auto.h > $@
	sed 's,	,    ,g;s/ *$$//;/^alias /d' $< | filter_verbatim >> $@

# end of $RCSfile: config.makefile,v $
#
