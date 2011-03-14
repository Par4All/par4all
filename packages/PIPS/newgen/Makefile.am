SUBDIRS=. src

# pkgconfig stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA= newgenlibs.pc

# aclocal stuff
ACLOCAL_AMFLAGS=-I makes/m4 -I src/gnulib/m4

include_HEADERS=config.h

# version stuff
VERSION_SH = $(top_srcdir)/makes/version.sh
.PHONY:newgen_version.h
BUILT_SOURCES=newgen_version.h
nodist_include_HEADERS=newgen_version.h
newgen_version.h:
	{ \
	  echo "#define NEWGEN_REV \"`$(VERSION_SH) $(top_srcdir)`\"" ; \
	  echo "#define NEWGEN_MAKE_REV \"`$(VERSION_SH) $(top_srcdir)/makes`\"" ; \
	} > $@
CLEANFILES	= newgen_version.h
