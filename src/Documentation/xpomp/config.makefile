#
# $Id$
#

DOC =		xpomp_manual.tex \
		xPOMP_window_explained.eps \
		xPOMP_window_explained.gif

SOURCES =	$(DOC)

INSTALL_DOC=	xpomp_manual.ps 
INSTALL_HTM=	xpomp_manual


DDOC =	$(INSTALL_DOC) \
	xpomp_manual.html \
	xpomp_manual/fractal.f \
	xpomp_manual/xPOMP_window_explained.gif

all: doc
doc: $(DDOC)

xpomp_manual/fractal.f: xpomp_manual.html 
	cp ../../Runtimes/xpomp/fractal.f xpomp_manual

xpomp_manual/xPOMP_window_explained.gif: 
	cp xPOMP_window_explained.gif xpomp_manual

clean: local-clean
local-clean:; $(RM) -r $(DDOC) xpomp_manual *.dvi

#
# end of pips makefile configuration file.
#
