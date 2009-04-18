#
# $Id$
#
# macros related to documentation generation.
#

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
# To publish on a WWW server:
RSYNC = rsync --archive --hard-links --delete --force --partial --compress --verbose

# whether to generate pdf directly from tex
ifdef use_pdflatex

%.pdf: %.tex
	-grep '\\makeindex' $*.tex && touch $*.ind
	$(PDFLTX) $<
	-grep '\\bibdata{' \*.aux && { $(BIBTEX) $* ; $(PDFLTX) $< ;}
	test ! -f $*.idx || { $(MAKEIDX) $*.idx ; $(PDFLTX) $< ;}
	$(PDFLTX) $<
	# Twice for the backref bibliography with hyperref:
	$(PDFLTX) $<
	touch $@

else # tex -> dvi -> ps -> pdf

# pdf (portable document format)
%.pdf: %.ps;	$(PS2PDF) $<

endif # use_pdflatex

# ps (post script)
%.ps: %.dvi;	$(DVIPS) $< -o

# latex
%.dvi: %.tex
	-grep '\\makeindex' $*.tex && touch $*.ind
	$(LATEX) $<
	-grep '\\bibdata{' \*.aux && { $(BIBTEX) $* ; $(LATEX) $< ;}
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
	cd $@; TEXINPUTS=.:..:: $(LX2HTM) $*
	# To have a hyperlinked index:
	cd $@; export TEXINPUTS=.:..:: ; if [ -r $*.idg ]; then \
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
make_destination_dir :

endif
