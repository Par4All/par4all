TARGET	= syntax
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=warning.h keywtbl.h tokyacc.h
dist_sysconf_DATA=f77keywords f77symboles
dist_noinst_DATA=gram.y

BUILT_SOURCES=$(TARGET).h keywtbl.h syn_yacc.y scanner.c

include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libsyntax.la

libsyntax_la_SOURCES=declaration.c \
		expression.c \
		equivalence.c \
		parser.c \
		procedure.c \
		reader.c \
		statement.c \
		return.c \
		malloc-info.c \
		macros.c \
		syn_yacc.y \
		scanner.l

AM_YFLAGS=-d -p syn_
AM_LFLAGS=-p syn_

syn_yacc.y: tokyacc.h gram.y
	$(AM_V_GEN)cat tokyacc.h $(srcdir)/gram.y > $@

keywtbl.h: warning.h f77keywords syn_yacc.h Makefile
	$(AM_V_GEN){\
		cat $(srcdir)/warning.h ; \
	  	echo "#include \"syn_yacc.h\"" ; \
	  	echo "static struct Skeyword keywtbl[] = {" ;\
	  	sed "s/^.*/{\"&\", TK_&},/" $(srcdir)/f77keywords ;\
	  	echo "{0, 0}" ;\
	  	echo "};" ; \
	} > $@

tokyacc.h: warning.h f77keywords f77symboles Makefile
	$(AM_V_GEN){ \
		cat $(srcdir)/warning.h ; \
		sed 's,\([^A-Z]*\)\(.*\),%token \1 TK_\2,' $(srcdir)/f77keywords $(srcdir)/f77symboles ;\
	} > $@

include $(srcdir)/../pipslibs_includes.mk
CLEANFILES=keywtbl.h syn_yacc.h syn_yacc.c tokyacc.h syn_yacc.y scanner.c XXXXXX.c

