#
# $Id$
# 

LIB_CFILES = \
	simdizer.c varwidth.c codegen.c unroll.c operatorid.c \
	treematch.c main.c

LIB_HEADERS = sac-local.h

LIB_OBJECTS = patterns.tab.o patterns.lex.o $(LIB_CFILES:.c=.o) 

%.lex.c: %.l %.tab.c
	flex -Cf -o$@ -Ppatterns_yy $<

%.tab.c: %.y
	bison -o $@ -d -p patterns_yy $<

patterns.tab.h: %.tab.c

clean: clean_patterns

clean_patterns:
	-$(RM) patterns.tab.c patterns.tab.h patterns.lex.c
