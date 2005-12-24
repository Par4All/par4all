# $Id$

TARGET	= top-level

LIB_CFILES =	dbm.c \
		help.c \
		logging.c \
		view.c \
		signal.c

INC_TARGET = $(TARGET).h

LIB_TARGET = lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
