# $Id$

INSTALL_DOC =	developer_guide.ps developer_guide.pdf

INSTALL_HTM = developer_guide

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
