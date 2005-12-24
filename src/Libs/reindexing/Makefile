# $Id$

TARGET	= reindexing

LIB_CFILES =	reindexing.c reindexing_utils.c prettyprint.c single_assign.c \
                bounds.c delay.c cell.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
