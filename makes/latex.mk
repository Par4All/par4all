SUFFIXES=.tex .pdf

.tex.pdf:
	$(PDFLATEX) $<
	$(PDFLATEX) $<
	$(PDFLATEX) $<

clean-local:
	$(RM) *.aux  *.idx  *.log  *.out *.toc
DISTCLEANFILES=$(doc_DATA)
