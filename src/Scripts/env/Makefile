# $Id$

ENV	=	pipsrc.sh pipsrc.csh

# is it still needed?
INSTALL_SHR	= $(ENV)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

#PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
pipsrc.sh: pipsrc.ref
	make-pipsrc.sh > $@
	chmod a-w $@

pipsrc.csh: pipsrc.sh
	make-pipsrc.csh < $< > $@
	chmod a-w $@

clean: local-clean

local-clean:
	$(RM) $(ENV)
