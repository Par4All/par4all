SUBDIRS=\
	misc newgen properties text-util pipsdbm \
	top-level ri-util conversion movements \
	comp_sections transformer bootstrap control flint \
	syntax c_syntax prettyprint \
	effects-generic effects-simple semantics complexity \
	continuation reductions effects-convex alias-classes \
	callgraph icfg ricedg \
	chains rice hyperplane transformations hwac expressions \
	locality instrumentation hpfc atomizer safescale sac phrase wp65 \
	preprocessor pipsmake step to_begin_with gpu scalopes static_controlize paf-util

BUILT_SOURCES=cprotoify

pips_includes:Makefile
	{ for dir in $(SUBDIRS) ; do echo -n "-I../$$dir -I../$(top_srcdir)/src/Libs/$$dir " ; done } > $@

cprotoify:Makefile pips_includes
	for dir in $(SUBDIRS) ; do $(MAKE) -C $${dir} $${dir}.h ; done


CLEANFILES=pips_includes

lib_LTLIBRARIES=libpipslibs.la

libpipslibs_la_SOURCES=
libpipslibs_la_LDFLAGS=$(NEWGENLIBS_LIBS) $(LINEARLIBS_LIBS)
libpipslibs_la_CPPFLAGS=$(NEWGENLIBS_CFLAGS) $(LINEARLIBS_CFLAGS)
libpipslibs_la_LIBADD=\
	top-level/libtop-level.la\
   	pipsmake/libpipsmake.la\
  	wp65/libwp65.la\
  	hpfc/libhpfc.la\
  	hyperplane/libhyperplane.la\
  	to_begin_with/libto_begin_with.la\
	instrumentation/libinstrumentation.la\
  	locality/liblocality.la\
  	expressions/libexpressions.la\
  	transformations/libtransformations.la\
  	hwac/libhwac.la\
	movements/libmovements.la\
  	bootstrap/libbootstrap.la\
  	callgraph/libcallgraph.la\
  	icfg/libicfg.la\
  	chains/libchains.la\
  	complexity/libcomplexity.la\
	conversion/libconversion.la\
  	prettyprint/libprettyprint.la\
  	atomizer/libatomizer.la\
  	syntax/libsyntax.la\
  	c_syntax/libc_syntax.la\
	effects-simple/libeffects-simple.la\
	effects-convex/libeffects-convex.la\
   	effects-generic/libeffects-generic.la\
   	alias-classes/libalias-classes.la\
	comp_sections/libcomp_sections.la\
  	semantics/libsemantics.la\
  	control/libcontrol.la\
  	continuation/libcontinuation.la\
  	rice/librice.la\
  	ricedg/libricedg.la\
	pipsdbm/libpipsdbm.la\
  	transformer/libtransformer.la\
  	preprocessor/libpreprocessor.la\
  	ri-util/libri-util.la\
	step/libstep.la\
	properties/libproperties.la\
	text-util/libtext-util.la\
   	misc/libmisc.la\
  	reductions/libreductions.la\
  	flint/libflint.la\
  	sac/libsac.la\
  	safescale/libsafescale.la\
  	phrase/libphrase.la\
	gpu/libgpu.la\
  	scalopes/libscalopes.la\
  	newgen/libnewgen.la \
	static_controlize/libstatic_controlize.la \
	paf-util/libpaf-util.la

