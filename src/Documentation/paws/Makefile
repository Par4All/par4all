# $Id$

use_pdflatex	= 1

ROOT	= ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.brf *.rel *.toc *.idx *.ilg *.ind *.log *.dvi \
		*.ps *.out *.lof *.pdf
