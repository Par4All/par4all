# $Id$

MANS =	Build.l \
	Delete.l \
	Display.l \
	Init.l \
	Perform.l \
	Pips.l \
	Select.l \
	epips.l \
	pips-cmd.l \
	tpips.l \
	wpips.l

HTMS =	$(MANS:.l=.html)

SOURCES = $(MANS) 

all: $(HTMS)

INSTALL_MAN=$(MANS)
INSTALL_HTM=$(HTMS)

clean: local-clean
local-clean:
	$(RM) $(HTMS)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk

