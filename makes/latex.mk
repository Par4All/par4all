SUFFIXES=.tex .pdf .eps .idraw 

.eps.pdf:
	$(EPSTOPDF) --outfile=$@ $<

.idraw.pdf:
	$(EPSTOPDF) --nogs --outfile=$@ $<

.tex.pdf:
	TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) $<
	TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) $<
	TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) $<

clean-local:
	rm -f *.aux  *.idx  *.log  *.out *.toc *.brf
DISTCLEANFILES=$(doc_DATA)
