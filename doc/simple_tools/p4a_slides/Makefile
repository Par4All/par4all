DOCUMENT = p4a_slides

# Where to install the documentation:
INSTALL_MACHINE_DIR=$(INSTALL_MACHINE_DOC_DIR)/simple_tools/p4a

# We have an index to compute
#MAKEINDEX=1

USE_PDFTEX=1

default: $(DOCUMENT)-expose.pdf

include $(P4A_ROOT)/doc/libs/par4all.mk
include $(P4A_ROOT)/doc/libs/tex/beamer.mk

publish:
	ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_MACHINE_DIR)
	rsync -av $(DOCUMENT)-expose.pdf $(INSTALL_MACHINE):$(INSTALL_MACHINE_DIR)

$(DOCUMENT).tex: p4a-help.tex

# How to generate the script documentation into slides:
%-help.tex: %.help Makefile
	../../../src/dev/optparse_help_to_tex --slides --no-output-usage $< > $@

%.help: ../../../src/simple_tools/%.py Makefile
	# Generate raw help message for the given command:
	$< -h > $@

clean::
	rm -rf *.help *-help.tex parsetab.py \
		*.aux *.bbl *.blg *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out *.help *-help.tex parsetab.py

### Local Variables:
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
