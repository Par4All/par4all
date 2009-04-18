# $Id$

use_pdflatex = 1
INSTALL_DOC = developer_guide.pdf
INSTALL_HTM = developer_guide.htdoc
# Hide the PDF into the HTML documentation directory
DOC.subd = $(INSTALL_HTM)

#PUBLISH_LOCATION = ssh.cri.ensmp.fr:~pips/public_html/developer_guide
#TO_BE_PUBLISH = $(INSTALL_DOC) $(INSTALL_HTM)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT	= pips

include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out
