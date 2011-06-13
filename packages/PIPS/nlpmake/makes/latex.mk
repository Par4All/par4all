SUFFIXES=.tex .pdf .eps .idraw 

.eps.pdf:
	$(AM_V_GEN)$(EPSTOPDF) --outfile=$@ $<

.idraw.pdf:
	$(AM_V_GEN)$(EPSTOPDF) --outfile=$@ $<

.tex.pdf:
	$(AM_V_GEN)TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) -interaction=batchmode $<
	$(AM_V_GEN)TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) -interaction=batchmode $<
	$(AM_V_GEN)TEXINPUTS=`$(KPSEPATH) tex`:$(builddir):$(srcdir) $(PDFLATEX) -interaction=batchmode $<

clean-local:
	rm -f *.aux  *.idx  *.log  *.out *.toc *.brf
DISTCLEANFILES=$(doc_DATA)
