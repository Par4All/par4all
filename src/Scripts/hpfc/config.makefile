#
# $RCSfile: config.makefile,v $ for hpfc scripts
# (version $Revision$, $Date: 1996/08/21 16:39:43 $, )
#

SCRIPTS = 	hpfc \
		hpfc_directives \
		hpfc_compile \
		hpfc_delete \
		hpfc_install

SOURCES	=	$(SCRIPTS) hpfc_interactive.c

INSTALL_SHR=	$(SCRIPTS)
INSTALL_BIN=	$(ARCH)/hpfc_interactive

all:	$(ARCH)/hpfc_interactive

#
# Some rules

$(ARCH)/hpfc_interactive: $(ARCH)/hpfc_interactive.o
	$(RM) $(ARCH)/hpfc_interactive
	$(LD) $(LDFLAGS) \
		-o $(ARCH)/hpfc_interactive \
		$(ARCH)/hpfc_interactive.o -lreadline -ltermcap
	chmod a+rx-w $(ARCH)/hpfc_interactive

clean: local-clean

local-clean:
	$(RM) $(ARCH)/hpfc_interactive.o $(ARCH)/hpfc_interactive *~

FC_HTML= /users/cri/coelho/public_html

web: hpfc_directives
	$(RM) $(FC_HTML)/hpfc_directives
	cp hpfc_directives $(FC_HTML)
	chmod a+r-wx $(FC_HTML)/hpfc_directives

# that is all
#
