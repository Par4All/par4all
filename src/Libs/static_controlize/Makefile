# $Id$

TARGET	= static_controlize

LIB_CFILES = \
	prettyprint.c loop_normalize.c static_controlize.c \
	utils.c forward_substitute.c

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
