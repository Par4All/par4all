# $Id$

TARGET	= control

LIB_CFILES = \
	clean_up_sequences.c \
	control.c \
	graph.c \
	hierarchize.c \
	module.c \
	reorder.c \
	unspaghettify.c \
	cfg.c \
	unreachable.c \
	bourdoncle.c

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
