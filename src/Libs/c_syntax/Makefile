# $Id$

TARGET	= c_syntax

YOPT	=	-d

INC_CFILES =	c_parser.c \
		statement.c \
		util.c

DERIVED_HEADERS = cyacc.h

DERIVED_CFILES = cyaccer.c clexer.c

LIB_CFILES = 	$(INC_CFILES) $(DERIVED_CFILES)

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

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
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS)

cyaccer.c cyacc.h: cyacc.y
	$(PARSE) cyacc.y
	sed 's/YY/C_/g;s/yy/c_/g' y.tab.c > cyaccer.c
	sed 's/YY/C_/g;s/yy/c_/g' y.tab.h > cyacc.h
	$(RM) y.tab.c y.tab.h

clexer.c: clex.l cyacc.h
	$(SCAN) clex.l | \
	sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/C_/g;s/yy/c_/g' > $@
