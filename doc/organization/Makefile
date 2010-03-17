INSTALL_DOC = par4all_organization.pdf
INSTALL_HTM = par4all_organization.htdoc

# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

use_pdflatex = 1
# Force HTML generation:
_HAS_HTLATEX_ = 1

# common stuff
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= pips
include $(ROOT)/makes/main.mk

default: full-compile phase7

publish:
	rsync -av par4all_organization.pdf par4all_organization.htdoc ridee.enstb.org:/var/www/par4all/doc/organization

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out
