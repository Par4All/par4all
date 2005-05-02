# $Id$

TARGET	= union

LIB_CFILES=	disjunct.c \
		path.c \
		reduc.c \
		sc_list.c

DERIVED_CFILES = sl_gram.c sl_lex.c  $(LIB_CFILES)

DERIVED_HEADERS = y.tab.h sl_lex.l sl_gram.y union-local.h

LIB_TARGET = lib$(TARGET).a

INC_TARGET = $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
include $(ROOT)/makes/main.mk

# ignore default rules
%.c: %.w; touch $@
%.h: %.w; touch $@
%.tex: %.w

sl_lex.c: sl_lex.l y.tab.h
	$(SCAN) $< | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/SL_/g;s/yy/slx_/g;'  > sl_lex.c

y.tab.h sl_gram.c: sl_gram.y
	$(PARSE) -d $< 
	sed 's/YY/SL_/g;s/yy/slx_/g' < y.tab.c > sl_gram.c
	$(RM) y.tab.c

# cancel rule
NUWEB	= /users/cri/coelho/tmp/nuweb/nuweb
code:;	$(NUWEB) -t UNION.w
doc:;	$(NUWEB) -f -o  UNION.w

clean: local-clean

local-clean:; $(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)
