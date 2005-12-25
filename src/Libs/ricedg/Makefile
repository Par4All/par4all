# $Id$

TARGET	= ricedg

LIB_CFILES =	util.c contexts.c testdep_util.c \
		ricedg.c prettyprint.c quick_privatize.c \
		trace.c impact_check.c

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
