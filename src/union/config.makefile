#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 18:10:56 $, 

# profiling for BC
# PROFILING=-pg -DNDEBUG -O3
CFLAGS+=	$(PROFILING)
LDFLAGS+=	$(PROFILING)

# sources
LIB_CFILES=	disjunct.c \
		path.c \
		reduc.c \
		sc_list.c

LIB_HEADERS=	union-type.h \
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

DEDUCED_CFILES=	sl_gram.c sl_lex.c
DEDUCED_HEADERS= y.tab.h y.tab.c

LIB_OBJECTS= $(LIB_CFILES:.c=.o) $(DEDUCED_CFILES:.c=.o)

union-local.h: union-types.h
	cp $< $@

sl_lex.c: sl_lex.l y.tab.h
	$(SCAN) $< | sed 's/YY/SLYY/g;s/yy/slyy/g' > $@

y.tab.h sl_gram.c: sl_gram.y
	$(PARSE) $< 
	sed 's/YY/SLYY/g;s/yy/slyy/g' < y.tab.c > sl_gram.c

# cancel rule
%.c:%.w

# for extracting web sources...
code:
	nuweb -t UNION.w

doc:
	nuweb -f -o  UNION.w

# end of $RCSfile: config.makefile,v $
#

