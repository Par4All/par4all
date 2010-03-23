SUFFIXES=.tex .pdf .eps .idraw 

.eps.pdf:
	$(EPSTOPDF) --outfile=$@ $<

.idraw.pdf:
	$(EPSTOPDF) --outfile=$@ $<

.tex.pdf:
	TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) $<
	TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) $<
	TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) $<

clean-local:
	$(RM) *.aux  *.idx  *.log  *.out *.toc *.brf
DISTCLEANFILES=$(doc_DATA)
