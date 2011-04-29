TARGET	= complexity
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libcomplexity.la
libcomplexity_la_SOURCES=\
	comp_scan.c comp_expr_to_pnome.c comp_unstr.c \
	comp_util.c comp_math.c comp_prettyprint.c polynome_ri.c \
	comp_matrice.c


include $(srcdir)/../pipslibs_includes.mk
EXTRA_DIST+=complexity_cost_tables
dist-hook:
	rm -rf `find $(distdir)/complexity_cost_tables -name .svn`
install-data-hook:
	test -d $(DESTDIR)$(sysconfdir)/complexity_cost_tables || mkdir $(DESTDIR)$(sysconfdir)/complexity_cost_tables
	chmod -R u+rw $(DESTDIR)$(sysconfdir)/complexity_cost_tables
	cp -r $(srcdir)/complexity_cost_tables $(DESTDIR)$(sysconfdir)/

uninstall-hook:
	chmod -R u+rw $(DESTDIR)$(sysconfdir)/complexity_cost_tables
	rm -rf $(DESTDIR)$(sysconfdir)/complexity_cost_tables
