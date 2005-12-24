# $Id$

TARGET	= flint

LIB_CFILES=	flint.c \
		flint_walk.c \
		flint_check.c \
		flint_utils.c \
		uninitialized_variables.c \
		type_checker.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
