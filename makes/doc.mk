#
# $Id$
#
# macros related to documentation generation.
#

LX2HTM	= htlatex
#L2HFLAGS= -link 8 -split 5 -local_icons
LATEX	= latex
BIBTEX	= bibtex
RMAN	= rman
MAKEIDX	= makeindex
DVIPS	= dvips
PS2PDF	= ps2pdf

%.pdf: %.ps;	$(PS2PDF) $<
%.ps: %.dvi;	$(DVIPS) $< -o
