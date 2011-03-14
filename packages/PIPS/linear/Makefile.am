SUBDIRS=. src

# pkgconfig stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA= linearlibs.pc

# aclocal stuff
ACLOCAL_AMFLAGS=-I makes/m4 -I src/gnulib/m4

# version stuff
VERSION_SH = $(top_srcdir)/makes/version.sh
.PHONY:linear_version.h
BUILT_SOURCES=linear_version.h
nodist_include_HEADERS=linear_version.h
linear_version.h:
	{ \
	  echo "#define LINEAR_REV \"`$(VERSION_SH) $(top_srcdir)`\"" ; \
	  echo "#define LINEAR_MAKE_REV \"`$(VERSION_SH) $(top_srcdir)/makes`\"" ; \
	} > $@
CLEANFILES	= linear_version.h
