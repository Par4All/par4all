# $Id$

TARGET	= reductions

LIB_CFILES =	stub.c \
		reductions.c \
		utils.c \
		transformation.c \
		prettyprint.c \
		call.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
