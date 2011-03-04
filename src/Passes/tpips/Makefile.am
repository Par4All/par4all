TARGET	= tpips
bin_PROGRAMS=tpips
tpips_SOURCES=tp_yacc.y tp_lex.l tpips.c wrapper.c revisions.c main_tpips.c
tpips_LDADD=../../Libs/libpipslibs.la
tpips_LDFLAGS=$(READLINE_LIBS) $(NCURSES_LIBS)\
	$(NEWGENLIBS_LIBS) $(LINEARLIBS_LIBS)
nodist_include_HEADERS=completion_list.h $(TARGET).h
dist_noinst_SCRIPTS=make_completion_lists.sh
include_HEADERS=pips_version.h


AM_YFLAGS=-d -p tp_
AM_LFLAGS=-p tp_

# hmmm... this version string is pretty useless
revisions.h:
	echo '#define PIPS_REV "$(PACKAGE_STRING) - $(PACKAGE_BUGREPORT)"' > $@

# regenerate everytime
.PHONY: pips_version.h
VERSION_SH = ../../../makes/version.sh

# added deps
$(ARCH)/revisions.o: pips_version.h
tpips.h: pips_version.h

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

BUILT_SOURCES=$(TARGET).h revisions.h completion_list.h tp_yacc.h
AM_CPPFLAGS=\
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
	$(READLINE_CFLAGS) $(NCURSES_CFLAGS)\
	$(NEWGENLIBS_CFLAGS) $(LINEARLIBS_CFLAGS)\
	-DNEWGEN_REV='"unknown"' \
	-DUTC_DATE='"unknown"' \
	-DLINEAR_REV='"unknown"' \
	-DNLPMAKE_REV='"unknown"' \
	-DCC_VERSION='"unknown"'

include $(top_srcdir)/makes/cproto.mk
DISTCLEANFILES=completion_list.h revisions.h
CLEANFILES=tp_yacc.[ch] tp_lex.c
