#
# $Id$
#

YFLAGS+=	-v -d

#
LIB_MAIN	=	main_tpips.c
LIB_CFILES	=	tpips.c
LIB_HEADERS	=	tpips-local.h \
			tp_lex.l \
			tp_yacc.y \
			build_completion_lists

DERIVED_HEADERS	= 	tp_yacc.h completion_list.h
DERIVED_CFILES	=	tp_yacc.c tp_lex.c
DERIVED_FILES	= 	y.output

# INC_CFILES	= 	$(DERIVED_FILES)

LIB_OBJECTS	=	$(LIB_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o)

#
TARGET_LIBS	= 	$(PIPS_LIBS) $(TPIPS_ADDED_LIBS)

tp_lex.c: tp_lex.l tp_yacc.h
	$(SCAN) tp_lex.l | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/TP_/g;s/yy/tp_/g' > tp_lex.c

tp_yacc.c tp_yacc.h: tp_yacc.y
	$(PARSE) tp_yacc.y
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
