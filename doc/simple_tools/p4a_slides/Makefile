DOCUMENT = p4a_slides

# Where to install the documentation:
INSTALL_MACHINE=ridee.enstb.org
INSTALL_DIR=/var/www/par4all/doc/simple_tools/p4a

# We have an index to compute
#MAKEINDEX=1

USE_PDFTEX=1

include ../../libs/tex/beamer.mk

publish:
	#ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_DIR)
	rsync -av $(DOCUMENT)-expose.pdf $(INSTALL_MACHINE):$(INSTALL_DIR)

$(DOCUMENT).tex: p4a-help.tex

# How to generate the script documentation into slides:
%-help.tex: %.help Makefile
	../../../src/dev/optparse_help_to_tex --slides $< > $@

%.help: ../../../src/simple_tools/% Makefile
	# Generate raw help message for the given command:
	$< -h > $@

clean::
	rm -rf *.help *-help.tex parsetab.py \
		*.aux *.bbl *.blg *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out *.help *-help.tex parsetab.py

### Local Variables:
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
