# $Id$

TARGET	= icfg

LIB_CFILES =	toposort.c icfg_scan.c print.c util.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
