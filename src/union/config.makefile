#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 2003/06/13 16:07:14 $, 

# profiling for BC
# PROFILING=-pg -DNDEBUG -O3
CFLAGS+=	$(PROFILING)
LDFLAGS+=	$(PROFILING)

# sources
LIB_CFILES=	disjunct.c \
		path.c \
		reduc.c \
		sc_list.c

LIB_HEADERS=	union-local.h \
		sl_lex.l \
		sl_gram.y \
		UNION.w \
		disjunct.w \
		path.w \
		reduc.w \
		sc_list.w \
		sl_io.w \
		union_archives.w \
		epsf_al.sty \
		fralgorithm.sty \
		leservot-fre.sty \
		leservot.sty

DERIVED_CFILES=	sl_gram.c sl_lex.c
DERIVED_HEADERS= y.tab.h 

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

LIB_OBJECTS= $(LIB_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o)

sl_lex.c: sl_lex.l y.tab.h
	$(SCAN) $< | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/SL_/g;s/yy/slx_/g;'  > sl_lex.c

y.tab.h sl_gram.c: sl_gram.y
	$(PARSE) -d $< 
	sed 's/YY/SL_/g;s/yy/slx_/g' < y.tab.c > sl_gram.c
	$(RM) y.tab.c

# cancel rule
%.c:%.w
%.h:%.w
%.tex:%.w

# for extracting web sources...
code:
	nuweb -t UNION.w

doc:
	nuweb -f -o  UNION.w

# end of $RCSfile: config.makefile,v $
#

