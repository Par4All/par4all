#
# $RCSfile: config.makefile,v $ for make
# $Date: 1996/09/03 12:58:50 $, 
#

SCRIPTS = 	make-pips \
		make_release \
		make_pips_release \
		make-gdbinit \
		install_pips

DOCS=		install.README \
		install.INSTALL

SOURCES	=	$(SCRIPTS) $(DOCS)

INSTALL_UTL=	$(SOURCES)

# that is all
#
