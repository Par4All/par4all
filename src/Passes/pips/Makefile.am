bin_PROGRAMS=pips
pips_SOURCES=pips.c revisions.c main_pips.c
pips_LDADD=../../Libs/libpipslibs.la
pips_LDFLAGS=$(NEWGENLIBS_LIBS) $(LINEARLIBS_LIBS)

TARGET	= pips
BUILT_SOURCES=$(TARGET).h revisions.h
include_HEADERS=$(TARGET).h revisions.h
include $(top_srcdir)/makes/cproto.mk
revisions.h:
	echo '#define PIPS_REV "$(PACKAGE_STRING) - $(PACKAGE_BUGREPORT)"' > $@

# work around a very "dumb" make
here   =  `[ "$(VPATH)" ] && echo $(VPATH) || echo '.'`

# regenerate everytime
.PHONY: pips_version.h
VERSION_SH = $(here)/../../../makes/version.sh

# added deps
$(ARCH)/revisions.o: pips_version.h
pips.h: pips_version.h

# hmmm... not sure about the portability...
pips_version.h:
	{ \
	  echo '#define PIPS_REV "$(shell $(VERSION_SH) $(here)/../../..)"' ; \
	  echo '#define PIPS_MAKE_REV "$(shell $(VERSION_SH) $(here)/../../../makes)"' ; \
	  echo '#define CC_VERSION "$(shell $(CC) --version | head -1)"' ; \
	  echo '#define UTC_DATE "$(shell date -u)"' ; \
	} > $@

AM_CPPFLAGS=\
	-I$(top_srcdir)/src/Documentation/constants \
	-I../../Documentation/pipsmake \
	-I../../Documentation/newgen \
	-I../../Libs/misc\
	-I../../Libs/newgen\
	-I../../Libs/ri-util\
	-I../../Libs/top-level\
	-I../../Libs/pipsmake\
	-I../../Libs/pipsdbm\
	-I../../Libs/properties\
	$(LINEARLIBS_CFLAGS)\
	$(NEWGENLIBS_CFLAGS)

CLEANFILES=revisions.h pips_version.h

