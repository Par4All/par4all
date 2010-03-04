TARGET	= tpips
bin_PROGRAMS=tpips
tpips_SOURCES=tp_yacc.y tp_lex.l tpips.c wrapper.c revisions.c main_tpips.c
tpips_LDADD=../../Libs/libpipslibs.la
tpips_LDFLAGS=$(READLINE_LIBS) $(TERMCAP_LIBS)
nodist_noinst_HEADERS=completion_list.h $(TARGET).h
dist_noinst_SCRIPTS=make_completion_lists.sh

AM_YFLAGS=-d -p tp_
AM_LFLAGS=-p tp_

revisions.h:
	echo '#define PIPS_REV "$(PACKAGE_STRING) - $(PACKAGE_BUGREPORT)"' > $@


# tpips completion

completion_list.h :	\
			make_completion_lists.sh \
			../../Documentation/pipsmake/phases.h \
			../../Documentation/pipsmake/resources.h \
			../../Documentation/pipsmake/printable_resources.h \
			../../Documentation/pipsmake/properties.rc
	sh $(srcdir)/make_completion_lists.sh \
		../../Documentation/pipsmake/phases.h \
		../../Documentation/pipsmake/resources.h \
		../../Documentation/pipsmake/printable_resources.h \
		../../Documentation/pipsmake/properties.rc \
		> $@

BUILT_SOURCES=$(TARGET).h revisions.h completion_list.h
AM_CPPFLAGS=\
	$(READLINE_CFLAGS) $(TERMCAP_CFLAGS)\
	$(NEWGENLIBS_CFLAGS) $(LINEARLIBS_CFLAGS)\
	-I$(top_srcdir)/src/Documentation/constants\
	-I../../Documentation/pipsmake\
	-I../../Documentation/newgen\
	-I../../Libs/newgen\
	-I../../Libs/misc\
	-I../../Libs/ri-util\
	-I../../Libs/pipsdbm\
	-I../../Libs/pipsmake\
	-I../../Libs/properties\
	-I../../Libs/top-level\
	-I../../Libs/preprocessor\
	-DNEWGEN_REV='"unknown"' \
	-DUTC_DATE='"unknown"' \
	-DLINEAR_REV='"unknown"' \
	-DNLPMAKE_REV='"unknown"' \
	-DCC_VERSION='"unknown"'
	
include $(top_srcdir)/makes/cproto.mk
CLEANFILES=completion_list.h revisions.h $(TARGET).h
