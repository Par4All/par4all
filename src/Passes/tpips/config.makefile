YFLAGS=		$(PIPS_YFLAGS) -v -d
LEX=		flex
#
TARGET_CFILES=	tpips.c
TARGET_HEADERS=	tpips-local.h ana_lex.l ana_syn.y
TARGET_OBJECTS=	$(TARGET_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o)
#
TARGET_LIBS= 	$(PIPS_LIBS) $(TPIPS_ADDED_LIBS)

DERIVED_HEADERS= y.tab.h completion_list.h
DERIVED_CFILES= y.tab.c lex.yy.c

ana_lex_completed.l: ana_lex.l $(PIPS_INCLUDEDIR)/resources.h $(PIPS_INCLUDEDIR)/phases.h
	$(PIPS_UTILDIR)/build_tpips_lex <ana_lex.l > ana_lex_completed.l

lex.yy.c: ana_lex_completed.l
	$(SCAN) ana_lex_completed.l | sed -e 's/YY/TP_/g;s/yy/tp_/g' > lex.yy.c

# on SunOS 4.1: yacc generates "extern char *malloc(), *realloc();"!
# filtred here.
y.tab.c: ana_syn.y
	$(YACC) $(YFLAGS) ana_syn.y
	sed -e '/extern char \*malloc/d;s/YY/TP_/g;s/yy/tp_/g' y.tab.c > m.tab.c
	mv m.tab.c y.tab.c
	sed -e 's/YY/TP_/g;s/yy/tp_/g' y.tab.h > m.tab.h
	mv m.tab.h y.tab.h

y.tab.h: y.tab.c

completion_list.h : $(PIPS_INCLUDEDIR)/resources.h $(PIPS_INCLUDEDIR)/phases.h
	$(PIPS_UTILDIR)/build_completion_lists > completion_list.h

depend: y.tab.c lex.yy.c

super-clean: clean
	rm -f y.tab.c lex.yy.c y.tab.h y.output ana_lex_completed.l

$(TARGET).h: $(DERIVED_HEADERS)
