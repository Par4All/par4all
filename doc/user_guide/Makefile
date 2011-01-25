INSTALL_DOC = par4all_user_guide.pdf
P4A_ROOT = ../..

all : $(INSTALL_DOC)

# Some own Par4All global config information:
include $(P4A_ROOT)/doc/libs/par4all.mk

%.dvi : %.tex $(SOURCES) Makefile
	-latex $(LATEXMODE) $*
	# Twice because of some BibTeXing...
	-bibtex $*
	-$(call COMPUTE_INDEX,$*)
	-makeindex $*
	-latex $(LATEXMODE) $*

%.ps : %.dvi Makefile
	dvips $* -o

%.pdf : ./%.ps ./Makefile
	ps2pdf $*.ps $*.pdf

clean:
	rm $(INSTALL_DOC) *.aux *.bbl *.blg *.log *.out
