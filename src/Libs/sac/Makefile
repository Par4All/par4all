# $Id$

TARGET	= sac

INC_CFILES = \
	simd_loop_const_elim.c if_conversion_compact.c if_conversion_init.c simd_loop_unroll.c simdizer.c varwidth.c codegen.c unroll.c operatorid.c \
	treematch.c atomizer.c vectransform.c reductions.c \
	singleass.c deatomizer.c if_conversion.c

DERIVED_CFILES = patterns.tab.c patterns.lex.c

DERIVED_HEADERS = patterns.tab.h

LIB_CFILES = $(INC_CFILES) $(DERIVED_CFILES)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

INSTALL_ETC = patterns.def

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
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
