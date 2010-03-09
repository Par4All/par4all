F_c=ri.c \
	dg.c \
	compsec.c \
	word_attachment.c \
	reductions_private.c \
	interval_graph.c \
	pipsdbm_private.c \
	eole_private.c \
	abc_private.c \
	alias_private.c \
	ubs_private.c \
	c_parser_private.c \
	complexity_ri.c \
	database.c \
	paf_ri.c \
	tiling.c \
	graph.c \
	parser_private.c \
	hpf_private.c \
	property.c \
	makefile.c \
	reduction.c \
	message.c \
	text.c \
	hpf.c \
	sac_private.c \
	ri_C.c \
	cloning.c \
	outlining_private.c \
	step_private.c \
	points_to_private.c \
	freia_spoc_private.c
# this duplication cannot be avoided
# for bsd-make compatibility ...
F_h=ri.h \
	dg.h \
	compsec.h \
	word_attachment.h \
	reductions_private.h \
	interval_graph.h \
	pipsdbm_private.h \
	eole_private.h \
	abc_private.h \
	alias_private.h \
	ubs_private.h \
	c_parser_private.h \
	complexity_ri.h \
	database.h \
	paf_ri.h \
	tiling.h \
	graph.h \
	parser_private.h \
	hpf_private.h \
	property.h \
	makefile.h \
	reduction.h \
	message.h \
	text.h \
	hpf.h \
	sac_private.h \
	ri_C.h \
	cloning.h \
	outlining_private.h \
	step_private.h \
	points_to_private.h \
	freia_spoc_private.h

dist_noinst_SCRIPTS=remove-latex-comments
noinst_LTLIBRARIES=libnewgenaux.la
libnewgenaux_la_SOURCES=$(F_c)
BUILT_SOURCES=specs.h $(F_c)
SUFFIXES=.newgen .tex
AM_CPPFLAGS=$(NEWGENLIBS_CFLAGS)

include_HEADERS=all_newgen_headers.h specs.h $(F_h)

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


CLEANFILES=$(F_c) $(include_HEADERS) *.newgen $(F_h) *.spec

