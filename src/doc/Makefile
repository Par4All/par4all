# $Id$
# Newgen documention

ETEX = 	newgen_manual.tex \
	newgen_paper.tex \
	tutoriel_newgen.tex

SOURCES = $(ETEX) obtention.txt

PS =	$(ETEX:%.tex=%.ps)

INSTALL_DOC =	$(PS)
INSTALL_HTM =   $(PS:%.ps=%)

ifdef NEWGEN_ROOT
ROOT	= $(NEWGEN_ROOT)
else
ROOT	= ../..
endif
include $(ROOT)/makes/base.mk

all: $(INSTALL_DOC)
ps: $(PS)
dvi: $(PS:.ps=.dvi)

clean: local-clean
local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) \
		$(PS:.ps=.html) *.dvi $(FTEX:.ftex=.tex) *.aux *.log
