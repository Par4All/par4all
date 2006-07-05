#
# $Id$
#
# macros related to documentation generation.
#

# Use the ultimate tool (well, IMHO (Ronan) :-) )
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

# If we want to generate HTML output from file.tex, create a directory "file"
# to hide junk details:
%: %.tex
	rm -rf $@
	mkdir $@
	# I guess we have kpathsea to deal with TEXINPUTS
	cd $@; TEXINPUTS=..:: $(LX2HTM) $@; ln -s $@.html index.html
