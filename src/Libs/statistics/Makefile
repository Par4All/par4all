# $Id$

TARGET	= statistics

LIB_CFILES = \
	stats.c \
	tiling_sequence.c \
	guard_elimination.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
