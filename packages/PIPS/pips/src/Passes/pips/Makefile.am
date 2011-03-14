bin_PROGRAMS=pips
pips_SOURCES=pips.c revisions.c main_pips.c
pips_LDADD=../../Libs/libpipslibs.la
pips_LDFLAGS=$(NEWGENLIBS_LIBS) $(LINEARLIBS_LIBS)

TARGET	= pips
BUILT_SOURCES=$(TARGET).h
include_HEADERS=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk

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

