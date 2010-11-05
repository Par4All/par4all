# Where to install the documentation:
INSTALL_MACHINE=download.par4all.org
INSTALL_MACHINE_DOC_DIR=/srv/www-par4all/download/doc

# Use specific styles:
TEXINPUTS:=$(P4A_ROOT)/doc/libs/tex//:$(TEXINPUTS)
export TEXINPUTS
