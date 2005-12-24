# $Id$

TARGET	= instrumentation

LIB_CFILES 	= \
	array_bound_check_bottom_up.c \
	array_bound_check_top_down.c \
	array_bound_check_instrumentation.c \
	array_bound_check_interprocedural.c \
	alias_propagation.c \
	alias_check.c \
	used_before_set.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
