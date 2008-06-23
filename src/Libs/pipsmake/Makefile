#
# $Id$
#

TARGET	= pipsmake
YOPT	= -d
INC_CFILES =	pipsmake.c \
		activate.c \
		callgraph.c \
		openclose.c \
		callback.c \
		unsplit.c \
		stubs.c

DERIVED_HEADERS = pipsmake_yacc.h
DERIVED_CFILES = pipsmake_yacc.c pipsmake_lex.c

LIB_CFILES =	$(DERIVED_CFILES)  $(INC_CFILES) 
INC_TARGET = $(TARGET).h
LIB_TARGET = lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local specials

clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)

$(TARGET).h: $(DERIVED_CFILES)

# pipsmake parser

YY2PM	= sed -e '/^FILE \*yyin/s/=[^,;]*//g;s/YY/PIPSMAKE_/g;s/yy/pipsmake_/g'

pipsmake_lex.c: readmakefile.l 
	$(SCAN) readmakefile.l | $(YY2PM) > $@

pipsmake_yacc.c: readmakefile.y
	$(PARSE) readmakefile.y
	$(YY2PM) y.tab.c > pipsmake_yacc.c
	$(YY2PM) y.tab.h > pipsmake_yacc.h
	$(RM) y.tab.[hc] y.output

pipsmake_yacc.h: pipsmake_yacc.c
