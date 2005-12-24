# $Id$

TARGET = effects-simple

LIB_CFILES = \
	binary_operators.c \
	interface.c \
	methods.c \
	interprocedural.c \
	prettyprint.c \
	unary_operators.c \
	filter_proper_effects.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
