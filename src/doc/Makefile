# $Id$
# Newgen documention

use_pdflatex	= 1

TEX = 	newgen_manual.tex \
	newgen_paper.tex \
	tutoriel_newgen.tex

SOURCES = $(TEX) obtention.txt

PDF 	= $(TEX:%.tex=%.pdf)

INSTALL_DOC =	$(PDF)

ifdef NEWGEN_ROOT
ROOT	= $(NEWGEN_ROOT)
else
ROOT	= ../..
endif
include $(ROOT)/makes/main.mk

all: $(INSTALL_DOC)

pdf: $(PDF)
ps: $(PS)
dvi: $(DVI)

clean: local-clean

local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) $(PDF) \
		*.html *.aux *.log *.ps *.dvi *.out *~
