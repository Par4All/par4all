#
# $RCSfile: config.makefile,v $ for dev
#

STF	=	stf \
		stf-workspace \
		stf-module

OTHERS	=	transformer-to-sc \
		directory-name

# scripts that must be chmod
SCRIPTS =	$(STF) $(OTHERS)

# source files of this directory
SOURCES	=	$(SCRIPTS)

# what to install and where
INSTALL_SHR=	$(STF)
INSTALL_UTL=	$(OTHERS)

# that is all
#
