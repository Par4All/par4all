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

# regenerate everytime
.PHONY: pips_version.h
VERSION_SH = ../../../makes/version.sh

# added deps
$(ARCH)/revisions.o: pips_version.h
pips.h: pips_version.h

# hmmm... not sure about the portability...
pips_version.h:
	{ \
	  echo '#define PIPS_REV "$(shell $(VERSION_SH) ../../..)"' ; \
	  echo '#define PIPS_MAKE_REV "$(shell $(VERSION_SH) ../../../makes)"' ; \
	  echo '#define CC_VERSION "$(shell $(CC) --version | head -1)"' ; \
	  echo '#define UTC_DATE "$(shell date -u)"' ; \
	} > $@

clean: clean-version
clean-version:
	$(RM) pips_version.h


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
	$(NEWGENLIBS_CFLAGS)\
	-DNEWGEN_REV='"unknown"' \
	-DUTC_DATE='"unknown"' \
	-DLINEAR_REV='"unknown"' \
	-DNLPMAKE_REV='"unknown"' \
	-DCC_VERSION='"unknown"'

CLEANFILES=revisions.h



