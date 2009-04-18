# $Id: Makefile 12428 2005-12-30 14:22:06Z coelho $

use_pdflatex	= 1

INSTALL_DOC =	tpips-user-manual.pdf
INSTALL_HTM = tpips-user-manual.htdoc
# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

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
	$(RM) $(INSTALL_DOC) *.aux *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.pdf *.out
