# Generate the documentation on the Par4All organization with many parts
# automatically generated

INSTALL_DOC = par4all_developer_guide.pdf
INSTALL_HTM = par4all_developer_guide.htdoc

# Where to install the documentation:
INSTALL_MACHINE_DIR=$(INSTALL_MACHINE_DOC_DIR)/developer_guide

# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

use_pdflatex = 1
# Force HTML generation:
_HAS_HTLATEX_ = 1

# common stuff, recycled directly from PIPS:
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= pips

# To be able to easily include sample scripts:
TEXINPUTS:=$(shell pwd)/../../src/simple_tools:$(TEXINPUTS):$(shell pwd)/../figures


# Some own Par4All global config information:
include $(P4A_ROOT)/doc/libs/par4all.mk

default: full-compile phase7

include $(ROOT)/makes/main.mk

# Where to find commands we want generate documentation:
VPATH = ../../src/dev:../../src/p4a_accel:../../src/postprocessor:../../src/simple_tools:../../src/validation

publish:
	ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_MACHINE_DIR)
	rsync -av $(INSTALL_DOC) $(INSTALL_HTM) $(INSTALL_MACHINE):$(INSTALL_MACHINE_DIR)

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out *.help *-help.tex parsetab.py release-notes.tex release-notes-extract.tex


# Add the dependence on some automatically generated help:
par4all_developer_guide.tex: p4a_setup-help.tex p4a_pack-help.tex p4a_post_processor.py-help.tex p4a_recover_includes-help.tex optparse_help_to_tex-help.tex p4a_validate-help.tex p4a_coffee-help.tex release-notes-extract.tex ../../src/simple_tools/p4a_setup_with_my_PIPS ../figures/p4a_work_flow.pdf  ../figures/p4a_work_flow.eps

# How to generate the script documentation:
%-help.tex: %.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 4 $< > $@

# p4a_pack-help is included at a deeper sectioning level:
p4a_pack-help.tex: p4a_pack.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 4 $< > $@

# p4a_validate-help is included at a deeper sectioning level:
p4a_validate-help.tex: p4a_validate.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 5 $< > $@

# p4a_coffee-help is included at a deeper sectioning level:
p4a_coffee-help.tex: p4a_coffee.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 4 $< > $@

# Match commands into the directories specified in the VPATH above
# If the command has a basic name:
%.help:% Makefile
	# Generate raw help message for the given command:
	$< -h > $@

# If the command has a basic.py name:
%.help:%.py Makefile
	# Generate raw help message for the given command:
	$< -h > $@

# Often, PDF output of tools (inkscape...) are bad because it is a whole
# page without cropping.  So rely on .eps output and apply translation and
# cropping:
%.pdf: %.eps
	ps2pdf $< $@
	pdfcrop $@
	mv $*-crop.pdf $@

# generate .eps from svg
%.eps: %.svg
	inkscape $< --export-eps=$@ --export-text-to-path

release-notes.tex: ../../RELEASE-NOTES.rst Makefile
	rst2latex --section-numbering $< $@

release-notes-extract.tex: release-notes.tex Makefile
	awk '/^This text is typeset according to the reStructuredText Markup/ { enable_output = 0 } { if (enable_output) print } /^% Document title/ { enable_output = 1 }' $< | \
	sed -e '/\\phantomsection%/d' -e 's/\\href{[^}]*}/\\url/g' \
	-e '/\\addcontentsline{/d'  -e 's/^\\section\*{/\\subsection{/' \
	-e '/\\author{/d' -e '/\\date{/d' -e '/^\\maketitle/d' \
	-e 's/^\\title{/\\section{/' > $@

### Local Variables:
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
