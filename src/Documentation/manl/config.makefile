#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/20 19:44:26 $, 

MANS =	Delete.l \
	Display.l \
	Init.l \
	Perform.l \
	Select.l \
	pips.l \
	wpips.l

HTMS =	$(MANS:.l=.html)

SOURCES = $(MANS) $(HTMS)

INSTALL_DOC_DIR=$(PIPS_ROOT)/Doc/manl

INSTALL_DOC=$(MANS)
INSTALL_HTM=$(HTMS)

# end of $RCSfile: config.makefile,v $
#
