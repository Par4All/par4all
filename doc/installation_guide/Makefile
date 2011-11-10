# Generate the documentation on the Par4All organization with many parts
# automatically generated

INSTALL_DOC = par4all_installation_guide.pdf
INSTALL_HTM = par4all_installation_guide.htdoc

# Where to install the documentation:
INSTALL_MACHINE_DIR=$(INSTALL_MACHINE_DOC_DIR)/installation_guide

# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

use_pdflatex = 1
# Force HTML generation:
_HAS_HTLATEX_ = 1

# common stuff, recycled directly from PIPS:
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= pips

# To be able to easily include sample scripts:
TEXINPUTS:=$(shell pwd)/../../src/simple_tools:$(TEXINPUTS)


# Some own Par4All global config information:
include $(P4A_ROOT)/doc/libs/par4all.mk

default: full-compile phase7

include $(ROOT)/makes/main.mk

# Where to find commands we want generate documentation:
VPATH = ../../src/dev:../../src/simple_tools

publish:
	ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_MACHINE_DIR)
	rsync -av $(INSTALL_DOC) $(INSTALL_HTM) $(INSTALL_MACHINE):$(INSTALL_MACHINE_DIR)

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out *.help *-help.tex parsetab.py


# Add the dependence on some automatically generated help:
par4all_installation_guide.tex: p4a_setup-help.tex 

# How to generate the script documentation:
%-help.tex: %.help Makefile
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

### Local Variables:
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
