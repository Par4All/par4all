#
# $RCSfile: config.makefile,v $ for hpfc scripts
#

CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS)
CPPFLAGS=	$(PIPS_CPPFLAGS)
LD=		$(PIPS_LD)
LDFLAGS=	$(PIPS_LDFLAGS)

SCRIPTS= 	hpfc \
		hpfc_directives \
		hpfc_add_includes \
		hpfc_filter \
		hpfc_add_warning \
		hpfc_generate_h \
		hpfc_compile \
		hpfc_generate_init \
		hpfc_delete \
		hpfc_install

FILES=
SFILES=		hpfc_interactive.c
RFILES=		hpfc_interactive

#
# Some rules

hpfc_interactive: hpfc_interactive.o
	$(RM) hpfc_interactive
	$(LD) $(LDFLAGS) \
		-o hpfc_interactive hpfc_interactive.o -lreadline -ltermcap
	chmod a-w hpfc_interactive

hpfc_interactive.o: hpfc_interactive.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c hpfc_interactive.c

clean:
	$(RM) hpfc_interactive.o hpfc_interactive *~

# that is all
#
