# $Id$

INSTALL_DOC =	developer_guide.ps

#INSTALL_HTM =

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(INSTALL_DOC) *.aux *.toc *.idx *.ilg *.ind *.log
