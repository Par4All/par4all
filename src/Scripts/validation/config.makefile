#
# $RCSfile: config.makefile,v $ for validation
#

SCRIPTS = 	Validate \
		validate-sequential \
		accept \
		bug-to-validate \
		dir-to-validate

SOURCES	=	$(SCRIPTS) config.makefile

INSTALL_UTL=	$(SCRIPTS)

# that is all
#
