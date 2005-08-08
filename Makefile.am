SUBDIRS = include Test

CPPFLAGS = -I$(srcdir)/include -Iinclude @CPPFLAGS@

ACLOCAL_AMFLAGS = -I m4

poly_src = $(srcdir)/applications
poly_arith = $(srcdir)/source/arith

CFILES= \
	source/arith/errors.c \
	source/kernel/errormsg.c \
	source/kernel/vector.c \
	source/kernel/matrix.c \
	source/kernel/polyhedron.c \
	source/kernel/polyparam.c \
	source/kernel/param.c \
	source/kernel/alpha.c \
	source/ehrhart/ehrhart.c \
	source/ehrhart/ext_ehrhart.c \
	source/ehrhart/eval_ehrhart.c \
	source/ehrhart/homogenization.c \
	source/ehrhart/ranking.c \
	source/kernel/matrix_addon.c \
	source/kernel/matrix_permutations.c \
	source/kernel/compress_parms.c \
	source/kernel/SolveDio.c \
	source/kernel/Lattice.c \
	source/kernel/Matop.c \
	source/kernel/NormalForms.c \
	source/kernel/Zpolyhedron.c		

sources = $(CFILES) \
	$(poly_arith)/assert.h \
	$(poly_arith)/arithmetique.h \
	$(poly_arith)/arithmetic_errors.h

EXTRA_DIST = doc source/oldpolytest.c

lib_LTLIBRARIES = @polylibs@
libpolylib32_la_LDFLAGS = -rpath $(libdir) -version-info @versioninfo@
libpolylib32_la_SOURCES = $(sources)
libpolylib32_la_CPPFLAGS = @CPPFLAGS32@
libpolylib64_la_LDFLAGS = -rpath $(libdir) -version-info @versioninfo@
libpolylib64_la_SOURCES = $(sources)
libpolylib64_la_CPPFLAGS = @CPPFLAGS64@
libpolylib128_la_LDFLAGS = -rpath $(libdir) -version-info @versioninfo@
libpolylib128_la_SOURCES = $(sources)
libpolylib128_la_CPPFLAGS = @CPPFLAGS128@
libpolylibgmp_la_LDFLAGS = -rpath $(libdir) -version-info @versioninfo@
libpolylibgmp_la_SOURCES = $(sources)
libpolylibgmp_la_CPPFLAGS = -DGNUMP

testlib_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
polytest_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
verif_ehrhart_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
Zpolytest_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
example_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
count_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
c2p_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
r2p_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
findv_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
pp_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
disjoint_union_sep_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
disjoint_union_adj_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
testehrhart_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
ehrhart_union_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
ehrhart_quick_apx_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
ehrhart_lower_bound_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
ehrhart_upper_bound_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)
ehrhart_ranking_CPPFLAGS = $(libpolylib@BITS@_la_CPPFLAGS)


EXTRA_LTLIBRARIES = libpolylib32.la libpolylib64.la libpolylib128.la \
		    libpolylibgmp.la

noinst_PROGRAMS = \
	testlib \
	polytest \
	verif_ehrhart \
	Zpolytest \
	example \
	count
bin_PROGRAMS = \
	c2p \
	r2p \
	findv \
	pp \
	disjoint_union_sep \
	disjoint_union_adj \
	testehrhart \
	ehrhart_union \
	ehrhart_quick_apx \
	ehrhart_upper_bound \
	ehrhart_lower_bound \
	ehrhart_ranking

LDADD = @polylib@
c2p_SOURCES = $(poly_src)/c2p.c
r2p_SOURCES = $(poly_src)/r2p.c
findv_SOURCES = $(poly_src)/findv.c
pp_SOURCES = $(poly_src)/pp.c
disjoint_union_sep_SOURCES = $(poly_src)/disjoint_union_sep.c
disjoint_union_adj_SOURCES = $(poly_src)/disjoint_union_adj.c
ehrhart_union_SOURCES = $(poly_src)/ehrhart_union.c
ehrhart_quick_apx_SOURCES = $(poly_src)/ehrhart_quick_apx.c
ehrhart_upper_bound_SOURCES =  $(poly_src)/ehrhart_upper_bound.c
ehrhart_lower_bound_SOURCES =  $(poly_src)/ehrhart_lower_bound.c
ehrhart_ranking_SOURCES =  $(poly_src)/ehrhart_ranking.c
testlib_SOURCES = $(poly_src)/testlib.c
polytest_SOURCES = $(poly_src)/polytest.c
example_SOURCES = $(poly_src)/example.c
count_SOURCES = source/count.c
Zpolytest_SOURCES = $(poly_src)/Zpolytest.c
verif_ehrhart_SOURCES = $(poly_src)/verif_ehrhart.c
testehrhart_SOURCES = $(poly_src)/testehrhart.c

longtest:
	(cd Test; $(MAKE) long_tests)

# Make the 'javadoc' style documentation
document:
	-mkdir doc
	$(DOXYGEN) polylib.doxygen

dist-hook: document
	rm -rf `find $(distdir)/doc -name CVS`
	cp -r doc/codeDoc $(distdir)/doc

install-data-local:
	@test -z "$(pkgconfig_libdir)" || $(mkdir_p) "$(DESTDIR)$(pkgconfig_libdir)"
	$(INSTALL_DATA) $(pkgconfig_libfile) "$(DESTDIR)$(pkgconfig_libdir)/$(pkgconfig_libfile)"

uninstall-local:
	rm -f "$(DESTDIR)$(pkgconfig_libdir)/$(pkgconfig_libfile)"
