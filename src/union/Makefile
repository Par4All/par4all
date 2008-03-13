# $Id$

TARGET	= union

CWEB_CFILES = \
		disjunct.c \
		path.c \
		reduc.c \
		sc_list.c

CWEB_OTHERS = \
		sl_lex.l \
		sl_gram.y \
		union-local.h

DERIVED_CFILES = \
		sl_gram.c \
		sl_lex.c

DERIVED_HEADERS = y.tab.h  

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

sl_lex.c: sl_lex.l y.tab.h
	$(SCAN) $< | \
	sed '/^FILE *\*/s,=[^,;]*,,g;s/YY/SL_/g;s/yy/slx_/g;'  > sl_lex.c

y.tab.h sl_gram.c: sl_gram.y
	$(PARSE) -d $< 
	sed 's/YY/SL_/g;s/yy/slx_/g' < y.tab.c > sl_gram.c
	$(RM) y.tab.c

clean: local-clean

local-clean:; $(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)
# If we were sure to have nuweb...
#local-clean:; $(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) $(CWEB_CFILE)
