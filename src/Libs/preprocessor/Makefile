# $Id$

TARGET	= preprocessor

INC_CFILES = \
	initializer.c \
	source_file.c \
	split_file.c \
	csplit_file.c

DERIVED_HEADERS = splitc.h

DERIVED_CFILES = splitcyaccer.c splitclexer.c

LIB_CFILES =	$(DERIVED_CFILES) $(INC_CFILES)

INC_TARGET = $(TARGET).h

LIB_TARGET = lib$(TARGET).a

YOPT	= -d

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) y.tab.c y.tab.h

splitcyaccer.c splitc.h: splitc.y
	$(PARSE) `pwd`/splitc.y
	sed 's/YY/SPLITC_/g;s/yy/splitc_/g' y.tab.c > splitcyaccer.c
	sed 's/YY/SPLITC_/g;s/yy/splitc_/g' y.tab.h > splitc.h
	$(RM) y.tab.c y.tab.h

splitclexer.c: splitc.l splitc.h
	$(SCAN) `pwd`/splitc.l | \
	sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/SPLITC_/g;s/yy/splitc_/g' > $@
