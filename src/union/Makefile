# $Id$

TARGET	= union

CWEB_CFILES =	disjunct.c \
		path.c \
		reduc.c \
		sc_list.c

CWEB_OTHERS = 	sl_lex.l \
		sl_gram.y \
		union-local.h

DERIVED_CFILES = \
		sl_gram.c \
		sl_lex.c

DERIVED_HEADERS = sl_gram.h

INC_CFILES = $(CWEB_CFILES)
LIB_CFILES = $(CWEB_CFILES) $(DERIVED_CFILES)

LIB_TARGET = lib$(TARGET).a
INC_TARGET = $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
PROJECT = linear
include $(ROOT)/makes/main.mk

# ignore default rules
%.c: %.w; touch $@
%.h: %.w; touch $@
%.tex: %.w

# can only work if nuweb executable is available
NUWEB	= nuweb
code:;	$(NUWEB) -t UNION.w
doc:;	$(NUWEB) -f -o  UNION.w

YY2SL	= sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/SL_/g;s/yy/slx_/g;'

sl_lex.c: sl_lex.l sl_gram.h
	$(SCAN) $< | $(YY2SL) > sl_lex.c

sl_gram.c: sl_gram.y
	$(PARSE) -d $< 
	$(YY2SL) y.tab.c > sl_gram.c
	$(YY2SL) y.tab.h > sl_gram.h
	$(RM) y.tab.c y.tab.h

sl_gram.h: sl_gram.c

clean: local-clean

local-clean:; $(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)
# If we were sure to have nuweb...
#local-clean:; $(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) $(CWEB_CFILE)
