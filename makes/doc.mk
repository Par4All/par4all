# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

# macros related to documentation generation.

# Use the ultimate tool (well, IMHO (Ronan) :-) )
LX2HTM	= htlatex
#L2HFLAGS= -link 8 -split 5 -local_icons
LATEX	= latex
PDFLTX	= pdflatex
BIBTEX	= bibtex
RMAN	= rman
MAKEIDX	= makeindex
DVIPS	= dvips
PS2PDF	= ps2pdf
EPS2PDF	= epstopdf --compress
# To publish on a WWW server:
RSYNC = rsync --archive --hard-links --delete --force --partial --compress --verbose --cvs-exclude

# whether to generate pdf directly from tex
ifdef use_pdflatex

%.pdf: %.tex
	-grep '\\makeindex' $*.tex && touch $*.ind
	$(PDFLTX) $<
	-grep '\\bibdata{' *.aux && { $(BIBTEX) $* ; $(PDFLTX) $< ;}
	test ! -f $*.idx || { $(MAKEIDX) $*.idx ; $(PDFLTX) $< ;}
	$(PDFLTX) $<
	# Twice for the backref bibliography with hyperref:
	$(PDFLTX) $<
	touch $@

# cleanup
clean: doc-clean-bibtex
doc-clean-bibtex:
	$(RM) *.bbl *.blg

else # tex -> dvi -> ps -> pdf

# pdf (portable document format)
%.pdf: %.ps;	$(PS2PDF) $<

endif # use_pdflatex

# ps (post script)
%.ps: %.dvi;	$(DVIPS) $< -o

%.eps: %.idraw
	cp $< $@

%.pdf: %.idraw
	$(EPS2PDF) --outfile=$@ $<

# latex
%.dvi: %.tex
	-grep '\\makeindex' $*.tex && touch $*.ind
	$(LATEX) $<
	-grep '\\bibdata{' *.aux && { $(BIBTEX) $* ; $(LATEX) $< ;}
	test ! -f $*.idx || { $(MAKEIDX) $*.idx ; $(LATEX) $< ;}
	$(LATEX) $<
	# Twice for the backref bibliography with hyperref:
	$(LATEX) $<
	touch $@

# If we want to generate HTML output from file.tex,
# create a "file.htdoc" directory to hide junk details:
%.htdoc: %.tex
	rm -rf $@
	mkdir $@
	# I guess we have kpathsea to deal with TEXINPUTS
	# Assume that an eventual index has been managed by the $(MAKEIDX) above.
	cd $@; TEXINPUTS=$(TEXINPUTS):.:..:: $(LX2HTM) $*
	# To have a hyperlinked index:
	cd $@; export TEXINPUTS=$(TEXINPUTS):.:..:: ; if [ -r $*.idg ]; then \
		tex "\def\filename{{$*}{idx}{4dx}{ind}} \input  idxmake.4ht" ;\
		$(MAKEIDX) -o $*.ind $*.4dx; fi ; \
	$(LX2HTM) $* ; \
	$(LX2HTM) $*
	# The document is displayed as the directory default view:
	ln -s $*.html $@/index.html

# Too dangerous (cf Documentation/web):
#clean:
#	$(RM) -rf $(INSTALL_HTM)

ifdef PUBLISH_LOCATION

publish: make_destination_dir
	$(RSYNC) $(TO_BE_PUBLISH) $(PUBLISH_LOCATION)

# Just to avoid publish to complaining if not implemented in the including
# Makefile:
make_destination_dir:

endif
