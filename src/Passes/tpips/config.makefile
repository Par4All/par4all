#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/23 09:27:35 $, 

YFLAGS+=	-v -d
#
LIB_CFILES=	tpips.c
LIB_HEADERS=	tpips-local.h \
		ana_lex.l \
		ana_syn.y \
		build_completion_lists \
		build_tpips_lex

DERIVED_HEADERS= tp_yacc.h completion_list.h
DERIVED_CFILES= tp_yacc.c tp_lex.c
DERIVED_FILES= ana_lex_completed.l

LIB_OBJECTS=	$(LIB_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o)

#
TARGET_LIBS= 	$(PIPS_LIBS) $(TPIPS_ADDED_LIBS)

#
#

ana_lex_completed.l:	ana_lex.l \
			$(PIPS_ROOT)/Include/resources.h \
			$(PIPS_ROOT)/Include/phases.h \
			$(PIPS_ROOT)/Share/properties.rc
	sh ./build_tpips_lex ana_lex.l > ana_lex_completed.l

# lex takes 100 times more time to process this file compared to flex
# (a few minutes versus a few seconds...).
tp_lex.c: ana_lex_completed.l tp_yacc.h
	$(SCAN) ana_lex_completed.l | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/TP_/g;s/yy/tp_/g' > tp_lex.c

# on SunOS 4.1: yacc generates "extern char *malloc(), *realloc();"!
# were filtred here.
tp_yacc.c tp_yacc.h: ana_syn.y
	$(PARSE) ana_syn.y
	sed 's/YY/TP_/g;s/yy/tp_/g' y.tab.c > tp_yacc.c
	sed -e 's/YY/TP_/g;s/yy/tp_/g' y.tab.h > tp_yacc.h
	$(RM) y.output y.tab.c y.tab.h

completion_list.h :	$(PIPS_ROOT)/Include/resources.h \
			$(PIPS_ROOT)/Include/phases.h \
			$(PIPS_ROOT)/Share/properties.rc
	sh ./build_completion_lists > completion_list.h

# for bootstraping the dependences...
tpips.h: completion_list.h

#
#
