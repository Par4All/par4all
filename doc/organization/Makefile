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

default: full-compile phase7

include $(ROOT)/makes/main.mk


publish:
	rsync -av par4all_organization.pdf par4all_organization.htdoc ridee.enstb.org:/var/www/par4all/doc/organization

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out *.help *-help.tex parsetab.py


# Add the dependence on some automatically generated help:
par4all_organization.tex: p4a_setup-help.tex p4a_pack-help.tex

# How to generate the script documentation:
%-help.tex: %.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 4 $< > $@

# p4a_pack-help is included at a deeper sectioning level:
p4a_pack-help.tex: p4a_pack.help Makefile
	../../src/dev/optparse_help_to_tex --article --article-section-level 5 $< > $@

%.help: ../../src/simple_tools/% Makefile
	# Generate raw help message for the given command:
	$< -h > $@

### Local Variables:
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
