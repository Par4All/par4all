bin_PROGRAMS=pips
pips_SOURCES=pips.c revisions.c main_pips.c
pips_LDADD=../../Libs/libpipslibs.la

TARGET	= pips
BUILT_SOURCES=$(TARGET).h revisions.h
include_HEADERS=$(TARGET).h revisions.h
include $(top_srcdir)/makes/cproto.mk
revisions.h:
	echo '#define PIPS_REV "$(PACKAGE_STRING) - $(PACKAGE_BUGREPORT)"' > $@

AM_CPPFLAGS=\
	$(LINEARLIBS_CFLAGS)\
	$(NEWGENLIBS_CFLAGS)\
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
	-DNEWGEN_REV='"unknown"' \
	-DUTC_DATE='"unknown"' \
	-DLINEAR_REV='"unknown"' \
	-DNLPMAKE_REV='"unknown"' \
	-DCC_VERSION='"unknown"'

MAINTAINERCLEANFILES=revisions.h



