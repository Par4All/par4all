# $Id$

TARGET	= paf-util

LIB_CFILES =	adg_read_paf.c bdt_read_paf.c utils.c print.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
