SUBDIRS=src
ACLOCAL_AMFLAGS=-I makes/m4 -I src/Libs/gnulib/m4
dist_noinst_HEADERS=link-warning.h
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA= pips.pc

# devel mode enables various checks
if WITH_DEVEL_MODE
check-includes:
	$(top_srcdir)/src/Scripts/dev/pipslibsdep.py -s $(top_srcdir) -b $(top_builddir)

inspect-symbols:
	for lib in src/Libs/* ; do \
		$(top_srcdir)/src/Scripts/dev/check_symbols.sh --check $$lib ; \
	done

check-properties:
	$(top_srcdir)/src/Scripts/dev/check_properties.sh src/Documentation/pipsmake/properties.rc $(top_srcdir)

endif
