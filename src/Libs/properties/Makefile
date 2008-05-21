# $Id$

TARGET	= properties

DERIVED_CFILES = properties.c

LIB_CFILES	= $(DERIVED_CFILES)

INC_TARGET	= $(TARGET).h

LIB_TARGET	= lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

properties.h: properties.c

properties.c: properties.l
	$(SCAN) $(abspath $<) | \
	sed -e '/^FILE \*yyin/s/=[^,;]*//g;s/YY/PROP_/g;s/yy/prop_/g;s/\(void *\*prop_alloc\)/static \1/;s/\(void *\*prop_realloc\)/static \1/;' \
		 > $@

clean: local-clean
local-clean:
	$(RM) $(DERIVED_CFILES)
