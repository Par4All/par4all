#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1997/03/17 12:59:00 $, 

MANS =	Build \
	Delete.l \
	Display.l \
	Init.l \
	Perform.l \
	Pips.l \
	Select.l \
	epips.l \
	pips.l \
	tpips.l \
	wpips.l

HTMS =	$(MANS:.l=.html)

SOURCES = $(MANS) 

all: $(HTMS)

INSTALL_DOC_DIR=$(PIPS_ROOT)/Doc/manl

INSTALL_DOC=$(MANS)
INSTALL_HTM=$(HTMS)

clean: local-clean
local-clean:
	$(RM) $(HTMS)

# end of $RCSfile: config.makefile,v $
#
