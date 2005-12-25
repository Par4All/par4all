# $Id$

TARGET	= comp_sections

LIB_CFILES =	comp_sections.c propagate.c ss.c operators.c myintrinsics.c \
		dbase.c prettyprint.c

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
