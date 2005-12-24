# $Id$

TARGET	= prgm_mapping

LIB_CFILES =	prgm_mapping.c broadcast.c utils.c print.c vvs.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
