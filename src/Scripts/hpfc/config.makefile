#
# $RCSfile: config.makefile,v $ for hpfc scripts
# (version $Revision$, $Date: 1997/03/10 08:52:48 $, )
#

SCRIPTS = 	hpfc \
		hpfc_directives \
		hpfc_compile \
		hpfc_delete \
		hpfc_install

SOURCES	=	$(SCRIPTS) hpfc_interactive.c hpfc_stubs.f

INSTALL_SHR=	$(SCRIPTS) hpfc_stubs.f hpfc_stubs.direct
INSTALL_BIN=	$(ARCH)/hpfc_interactive

all:	$(ARCH)/hpfc_interactive hpfc_stubs.direct

#
# Some rules

$(ARCH)/hpfc_interactive: $(ARCH)/hpfc_interactive.o
	$(RM) $(ARCH)/hpfc_interactive
	$(LD) $(LDFLAGS) \
		-o $(ARCH)/hpfc_interactive \
		$(ARCH)/hpfc_interactive.o -lreadline -ltermcap
	chmod a+rx-w $(ARCH)/hpfc_interactive

# the direct version of the stubs need not be filtered by hpfc_directives.
hpfc_stubs.direct: hpfc_stubs.f
	# building $@ from $<
	sed 's,^!fcd\$$ fake,      call hpfc9,' $< > $@

clean: local-clean

local-clean:
	$(RM) $(ARCH)/hpfc_interactive.o $(ARCH)/hpfc_interactive \
		*~ hpfc_stubs.direct

FC_HTML= /users/cri/coelho/public_html

web: hpfc_directives
	$(RM) $(FC_HTML)/hpfc_directives
	cp hpfc_directives $(FC_HTML)
	chmod a+r-wx $(FC_HTML)/hpfc_directives

# that is all
#
