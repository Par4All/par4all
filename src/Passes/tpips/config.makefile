#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/08 14:05:46 $, 

LEX=		flex
YFLAGS+=	-v -d
#
LIB_CFILES=	tpips.c
LIB_HEADERS=	tpips-local.h ana_lex.l ana_syn.y
LIB_OBJECTS=	$(LIB_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o)
#
TARGET_LIBS= 	$(PIPS_LIBS) $(TPIPS_ADDED_LIBS)

DERIVED_HEADERS= y.tab.h completion_list.h
DERIVED_CFILES= y.tab.c lex.yy.c
DERIVED_FILES= y.output ana_lex_completed.l

ana_lex_completed.l:	ana_lex.l \
			$(PIPS_INCLUDEDIR)/resources.h \
			$(PIPS_INCLUDEDIR)/phases.h \
			$(PIPS_LIBDIR)/properties.rc
	$(PIPS_UTILDIR)/build_tpips_lex ana_lex.l > ana_lex_completed.l

lex.yy.c: ana_lex_completed.l y.tab.h
	$(SCAN) ana_lex_completed.l | sed -e 's/YY/TP_/g;s/yy/tp_/g' > lex.yy.c

# on SunOS 4.1: yacc generates "extern char *malloc(), *realloc();"!
# filtred here.
y.tab.c y.tab.h: ana_syn.y
	$(PARSE) ana_syn.y
	sed -e '/extern char \*malloc/d;s/YY/TP_/g;s/yy/tp_/g' y.tab.c > m.tab.c
	mv m.tab.c y.tab.c
	sed -e 's/YY/TP_/g;s/yy/tp_/g' y.tab.h > m.tab.h
	mv m.tab.h y.tab.h

completion_list.h :	$(PIPS_INCLUDEDIR)/resources.h \
			$(PIPS_INCLUDEDIR)/phases.h \
			$(PIPS_LIBDIR)/properties.rc
	$(PIPS_UTILDIR)/build_completion_lists > completion_list.h

# for bootstraping the dependences...
tpips.o: completion_list.h
tpips.c: completion_list.h
