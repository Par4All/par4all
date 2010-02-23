F_c=cloning.c property.c step_private.c graph.c message.c freia_spoc_private.c eole_private.c text.c hpf.c sac_private.c reduction.c dcdg.c compsec.c interval_graph.c dg.c alias_private.c database.c tiling.c paf_ri.c outlining_private.c word_attachment.c ubs_private.c hpf_private.c makefile.c parser_private.c complexity_ri.c optimizer.c ri.c reductions_private.c ri_C.c abc_private.c pipsdbm_private.c c_parser_private.c

dist_noinst_SCRIPTS=remove-latex-comments
noinst_LTLIBRARIES=libnewgenaux.la
libnewgenaux_la_SOURCES=$(F_c)
BUILT_SOURCES=specs.h $(F_c)
SUFFIXES=.newgen .tex
AM_CPPFLAGS=$(NEWGENLIBS_CFLAGS)

include_HEADERS=all_newgen_headers.h specs.h

all_newgen_headers.h: specs.h Makefile
	sed -n 's,^\(.*\)_spec.*,#include "\1.h",p' specs.h > $@

specs.h $(F_spec):Makefile $(srcdir)/*.tex
	$(MAKE) `echo $(F_c) | sed -e 's/\.c/.newgen /g'`
	PATH=`dirname $(NEWGEN)`:$$PATH $(NEWGEN) -c `echo $(F_c) | sed -e 's/\.c/.newgen/g'` | $(MAKE_ALL_SPECS) > specs.h

.tex.newgen:
	$(srcdir)/remove-latex-comments $<

# this is a work-around for the lack of support for patsubst in automake
# for portability reasons
dist-hook:
	cp $(srcdir)/*.tex $(distdir)

install-data-hook:
	$(MAKE) `echo $(F_c) | sed -e 's/\.c/.h/g'`
	for ax_file in `echo $(F_c) | sed -e 's/\.c/.h/g'` ; do $(INSTALL_HEADER) $$ax_file $(DESTDIR)$(includedir)/$$ax_file ; done

uninstall-hook:
	for ax_file in `echo $(F_c) | sed -e 's/\.c/.h/g'` ; do rm -f $(DESTDIR)$(includedir)/$$ax_file ; done



CLEANFILES=$(F_c) $(include_HEADERS) *.newgen *.h *.spec

