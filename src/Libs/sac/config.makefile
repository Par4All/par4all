#
# $Id$
# 

LIB_CFILES = \
	simdizer.c varwidth.c codegen.c unroll.c operatorid.c \
	treematch.c atomizer.c main.c vectransform.c reductions.c \
	singleass.c

LIB_HEADERS = sac-local.h patterns.l patterns.y patterns.def

DERIVED_HEADERS= patterns.tab.h
DERIVED_CFILES= patterns.tab.c patterns.lex.c

LIB_OBJECTS = $(DERIVED_CFILES:.c=.o) $(LIB_CFILES:.c=.o)

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

patterns.lex.c: patterns.l patterns.tab.h
	flex -Cf -opatterns.lex.c -Ppatterns_yy patterns.l

patterns.tab.c patterns.tab.h: patterns.y
	bison -o patterns.tab.c -d -p patterns_yy patterns.y
