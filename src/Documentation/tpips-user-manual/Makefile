# $Id: Makefile 12428 2005-12-30 14:22:06Z coelho $

INSTALL_DOC =	tpips-user-manual.ps

#INSTALL_HTM = tpips-user-manual.htdoc

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT	= pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(INSTALL_DOC) *.aux *.toc *.idx *.ilg *.ind *.log
