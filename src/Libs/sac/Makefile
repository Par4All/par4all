# $Id$

TARGET	= sac

INC_CFILES = \
	simdizer.c varwidth.c codegen.c unroll.c operatorid.c \
	treematch.c atomizer.c vectransform.c reductions.c \
	singleass.c

DERIVED_CFILES = patterns.tab.c patterns.lex.c

DERIVED_HEADERS = patterns.tab.h

LIB_CFILES = $(INC_CFILES) $(DERIVED_CFILES)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) $(DERIVED_HEADERS) $(DERIVED_CFILES) 

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

patterns.lex.c: patterns.l patterns.tab.h
	flex -Cf -opatterns.lex.c -Ppatterns_yy patterns.l

patterns.tab.c patterns.tab.h: patterns.y
	bison -o patterns.tab.c -d -p patterns_yy patterns.y
