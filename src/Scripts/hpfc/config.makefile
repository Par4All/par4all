#
# $RCSfile: config.makefile,v $ for hpfc scripts
# (version $Revision$, $Date: 1996/08/21 15:36:27 $, )
#

SHR_SCRIPTS = 	hpfc \
		hpfc_directives \
		hpfc_compile \
		hpfc_delete \
		hpfc_install

RTM_SCRIPTS =	hpfc_llcmd \
		hpfc_add_warning \
		hpfc_generate_h \
		hpfc_generate_init

SCRIPTS = 	$(SHR_SCRIPTS) $(RTM_SCRIPTS)
SOURCES	=	$(SCRIPTS) hpfc_interactive.c

INSTALL_SHR=	$(SHR_SCRIPTS)
INSTALL_BIN=	$(ARCH)/hpfc_interactive
INSTALL_RTM=	$(RTM_SCRIPTS)

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
