# $Id$
# Newgen documention

TEX = 	newgen_manual.tex \
	newgen_paper.tex \
	tutoriel_newgen.tex

SOURCES = $(TEX) obtention.txt

DVI	= $(TEX:%.tex=%.dvi)
PS	= $(TEX:%.tex=%.ps)
PDF 	= $(TEX:%.tex=%.pdf)

INSTALL_DOC =	$(PDF)
#INSTALL_HTM =   $(PS:%.ps=%)

ifdef NEWGEN_ROOT
ROOT	= $(NEWGEN_ROOT)
else
ROOT	= ../..
endif
include $(ROOT)/makes/base.mk

all: $(INSTALL_DOC)

pdf: $(PDF)
ps: $(PS)
dvi: $(DVI)

clean: local-clean

local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) $(PDF) $(PS) $(DVI) \
		*.html *.aux *.log *~
