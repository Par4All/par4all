DOCUMENT = p4a_slides

# Where to install the documentation:
INSTALL_MACHINE=ridee.enstb.org
INSTALL_DIR=/var/www/par4all/doc/simple_tools/p4a

# We have an index to compute
#MAKEINDEX=1

USE_PDFTEX=1

# OK, not really usable except by me right now... :-)
# point to where our local TeX stuff is installed:
TEX_ROOT=/home/keryell/projets/Wild_Systems/svn/doc/TeX

# Add this to the TeX path:
INSERT_TEXINPUTS=::$(TEX_ROOT)//:$(TEX_ROOT)/../Images//
#APPEND_TEXINPUTS=$(TEX_ROOT)//:$(TEX_ROOT)/../Images//
include $(TEX_ROOT)/Makefiles/beamer.mk

publish:
	ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_DIR)
	rsync -av $(DOCUMENT)-expose.pdf $(INSTALL_MACHINE):$(INSTALL_DIR)

$(DOCUMENT).tex: p4a.help.tex

p4a.help.tex: p4a.help
	# Compile the help output into slides:
	../../../src/dev/optparse_help_to_tex $< > $@

.PHONY: always_rebuild

p4a.help: always_rebuild
	p4a -h > $@
