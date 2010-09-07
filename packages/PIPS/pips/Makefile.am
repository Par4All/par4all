SUBDIRS=src
ACLOCAL_AMFLAGS=-I makes/m4 -I src/Libs/gnulib/m4
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

cscope:
	find $(top_srcdir)/src -name '*.[chly]' -print >  cscope.files && \
	find $(top_builddir)/src/Documentation -name '*.h' -print >>  cscope.files && \
	cscope -b && \
	rm -f cscope.files
endif

MAINTAINERCLEANFILES=cscope.out
