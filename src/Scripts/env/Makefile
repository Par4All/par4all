# $Id$

INSTALL_SHR	= pipsrc.sh pipsrc.csh

INSTALL_UTL	= sh2csh.pl

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

#PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
pipsrc.csh: pipsrc.sh
	sh2csh.pl $< > $@

clean: local-clean

local-clean:
	$(RM) pipsrc.csh
