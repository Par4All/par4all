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

# pdf (portable document format)
%.pdf: %.ps;	$(PS2PDF) $<

# ps (post script)
%.ps: %.dvi;	$(DVIPS) $< -o

# latex
%.dvi: %.tex
	-grep '\\makeindex' $*.tex && touch $*.ind
	$(LATEX) $<
	-grep '\\bibdata{' \*.aux && { $(BIBTEX) $* ; $(LATEX) $< ;}
	test ! -f $*.idx || { $(MAKEIDX) $*.idx ; $(LATEX) $< ;}
	$(LATEX) $<
	touch $@

# If we want to generate HTML output from file.tex, 
# create a "file.htdoc" directory to hide junk details:
%.htdoc: %.tex
	rm -rf $@
	mkdir $@
	# I guess we have kpathsea to deal with TEXINPUTS
	cd $@; TEXINPUTS=..:: $(LX2HTM) $*; ln -s $*.html index.html
