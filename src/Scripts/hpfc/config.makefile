#
# $RCSfile: config.makefile,v $ for hpfc scripts
#

SCRIPTS= 	hpfc \
		hpfc_directives \
		hpfc_add_warning \
		hpfc_generate_h \
		hpfc_compile \
		hpfc_generate_init \
		hpfc_delete \
		hpfc_install \
		hpfc_llcmd

FILES=
SFILES=		hpfc_interactive.c
RFILES=		$(ARCH)/hpfc_interactive

#
# Some rules

$(ARCH)/hpfc_interactive: $(ARCH)/hpfc_interactive.o
	$(RM) $(ARCH)/hpfc_interactive
	$(LD) $(LDFLAGS) \
		-o $(ARCH)/hpfc_interactive \
		$(ARCH)/hpfc_interactive.o -lreadline -ltermcap
	chmod a-w $(ARCH)/hpfc_interactive

clean: local-clean

local-clean:
	$(RM) $(ARCH)/hpfc_interactive.o $(ARCH)/hpfc_interactive *~

web: hpfc_directives
	$(RM) $(HOME)/public_html/hpfc_directives
	cp hpfc_directives $(HOME)/public_html/
	chmod a+r $(HOME)/public_html/hpfc_directives

# that is all
#
