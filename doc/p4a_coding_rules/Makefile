INSTALL_DOC = p4a_coding_rules.pdf
INSTALL_HTM = p4a_coding_rules.htdoc

# Where to install the documentation:
INSTALL_MACHINE_DIR=$(INSTALL_MACHINE_DOC_DIR)/p4a_coding_rules
PUBLISH_LOCATION=$(INSTALL_MACHINE):$(INSTALL_MACHINE_DIR)
TO_BE_PUBLISH=$(INSTALL_DOC) $(INSTALL_HTM)

TEXINPUTS:=$(TEXINPUTS):$(shell pwd)/../../figures//::$(shell pwd)/../../../examples/Python::

# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

use_pdflatex = 1
# Force HTML generation:
_HAS_HTLATEX_ = 1

# common stuff
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= pips

# Some global config information:
include $(P4A_ROOT)/doc/libs/par4all.mk

default: full-compile phase7

include $(ROOT)/makes/main.mk


make_destination_dir :
	ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_MACHINE_DIR)

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out *.help *-help.tex parsetab.py


### Local Variables:
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
