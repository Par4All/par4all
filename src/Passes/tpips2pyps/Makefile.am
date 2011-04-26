TARGET	= tpips2pyps
bin_PROGRAMS=tpips2pyps
tpips2pyps_SOURCES=tp_yacc.y tp_lex.l tpips2pyps.c main_tpips.c
tpips2pyps_LDADD=../../Libs/libpipslibs.la
tpips2pyps_LDFLAGS=$(READLINE_LIBS) $(NCURSES_LIBS)\
	$(NEWGENLIBS_LIBS) $(LINEARLIBS_LIBS)



AM_YFLAGS=-d -p tp_
AM_LFLAGS=-p tp_

BUILT_SOURCES=$(TARGET).h tp_yacc.h loop_phases.h
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
	$(NEWGENLIBS_CFLAGS) $(LINEARLIBS_CFLAGS)

p2p_cmd= $(srcdir)/pipsmakerc-extract-loop-phase.py $(top_srcdir)/src/Documentation/pipsmake/pipsmake-rc.tex ../../Documentation/pipsmake/properties.rc ../../Documentation/pipsmake/pipsdep.rc
loop_phases.h:$(p2p_cmd) Makefile
	$(PYTHON) $(p2p_cmd) > loop_phases.h


include $(top_srcdir)/makes/cproto.mk
CLEANFILES=tp_yacc.[ch] tp_lex.c loop_phases.h
