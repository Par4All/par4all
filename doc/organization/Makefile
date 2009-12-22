# Temporary version.
# To compile try something like
#  make _HAS_HTLATEX_=1 ROOT=~/projets/Wild_Systems/PIPS/git-svn-work/pips full-compile
# rsync -a par4all_organization.pdf par4all_organization.htdoc ridee.enstb.org:/var/www/par4all/doc/organization
use_pdflatex = 1
INSTALL_DOC = par4all_organization.pdf
INSTALL_HTM = par4all_organization.htdoc
# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

# common stuff
ROOT    = ../../..
PROJECT	= pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out
