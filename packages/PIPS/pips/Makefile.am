# visit . first for built_sources generation
SUBDIRS=. src
ACLOCAL_AMFLAGS=-I makes/m4 -I src/Libs/gnulib/m4
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA= pips.pc

# version stuff
VERSION_SH = $(top_srcdir)/makes/version.sh
.PHONY:pips_version.h
BUILT_SOURCES=pips_version.h
nodist_include_HEADERS=pips_version.h
EXTRA_DIST=makes/version.sh

pips_version.h:
	{ \
	  echo "#define PIPS_REV \"`$(VERSION_SH) $(top_srcdir)`\"" ; \
	  echo "#define PIPS_MAKE_REV \"`$(VERSION_SH) $(top_srcdir)/makes`\"" ; \
	  echo "#define CC_VERSION \"`$(CC) --version | head -1`\"" ; \
	  echo "#define UTC_DATE \"`date -u`\"" ; \
	} > $@

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
CLEANFILES=pips_version.h

# do not package svn files
dist-hook:
	rm -rf `find $(distdir) -name .svn`
