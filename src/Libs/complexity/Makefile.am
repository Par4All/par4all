TARGET	= complexity
nodist_include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libcomplexity.la
dist_noinst_DATA=complexity_cost_tables.tar.gz
libcomplexity_la_SOURCES=\
	comp_scan.c comp_expr_to_pnome.c comp_unstr.c \
	comp_util.c comp_math.c comp_prettyprint.c polynome_ri.c \
	comp_matrice.c

complexity_cost_tables.tar.gz:
	if test ! -d complexity_cost_tables && test -d $(srcdir)/complexity_cost_tables ; then \
		$(LN_S) $(srcdir)/complexity_cost_tables . ; \
	fi
	if test -r $(srcdir)/complexity_cost_tables.tar.gz ; then \
		$(LN_S) $(srcdir)/complexity_cost_tables.tar.gz . ; \
	else \
		tar czf $@ --exclude="*.svn*" complexity_cost_tables ; \
	fi

include $(srcdir)/../pipslibs_includes.mk
install-data-hook:complexity_cost_tables.tar.gz
	rm -rf complexity_cost_tables && \
		tar xzf `( test -f complexity_cost_tables.tar.gz && echo complexity_cost_tables.tar.gz ) || echo $(srcdir)/complexity_cost_tables.tar.gz ` && \
		cp -r complexity_cost_tables $(DESTDIR)$(sysconfdir)/ && rm -rf complexity_cost_tables

uninstall-hook:
	rm -rf $(DESTDIR)$(sysconfdir)/complexity_cost_tables
