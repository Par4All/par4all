# $Id$

TARGET 	= text-util

LIB_CFILES	= text_print.c util.c

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
