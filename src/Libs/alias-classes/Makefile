# $Id$

TARGET	= alias-classes

LIB_CFILES =	alias_pairs.c \
		alias_lists.c \
		alias_classes.c \
		prettyprint.c

INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
