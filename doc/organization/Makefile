# Generate the documentation on the Par4All organization with many parts
# automatically generated

INSTALL_DOC = par4all_organization.pdf
INSTALL_HTM = par4all_organization.htdoc

# Where to install the documentation:
INSTALL_MACHINE_DIR=$(INSTALL_MACHINE_DOC_DIR)/organization

# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

use_pdflatex = 1
# Force HTML generation:
_HAS_HTLATEX_ = 1

# common stuff, recycled directly from PIPS:
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= pips

# Some own Par4All stuff:
include $(P4A_ROOT)/doc/libs/par4all.mk

default: full-compile phase7

include $(ROOT)/makes/main.mk


publish:
	ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_MACHINE_DIR)
	rsync -av $(INSTALL_DOC) $(INSTALL_HTM) $(INSTALL_MACHINE):$(INSTALL_MACHINE_DIR)

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out *.help *-help.tex parsetab.py release-notes.tex release-notes-extract.tex


# Add the dependence on some automatically generated help:
par4all_organization.tex: p4a_setup-help.tex p4a_pack-help.tex p4a_post_processor.py-help.tex p4a_recover_includes-help.tex optparse_help_to_tex-help.tex p4a_validate-help.tex release-notes-extract.tex

# How to generate the script documentation:
%-help.tex: %.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 4 $< > $@

# p4a_pack-help is included at a deeper sectioning level:
p4a_pack-help.tex: p4a_pack.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 5 $< > $@

# Where to find commands we want generate documentation:
VPATH = ../../src/dev:../../src/p4a_accel:../../src/postprocessor:../../src/simple_tools:../../src/validation

# Match commands into the directories specified in the VPATH above
%.help:% Makefile
	# Generate raw help message for the given command:
	$< -h > $@

release-notes.tex: ../../RELEASE-NOTES.txt Makefile
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
